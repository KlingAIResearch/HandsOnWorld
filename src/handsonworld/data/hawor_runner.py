"""Adapter for an independently installed, unmodified HaWoR checkout.

HandsOnWorld enumerates candidates and translates the shared estimator protocol.
Model loading, MANO geometry/topology, rendering, SLAM, canonical interpolation
and learned infilling are provided by the external installation. No extra adapter
file or edits inside that checkout are needed. Run one checkout per process:
upstream imports and resource paths use process-global module names and cwd.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Iterable

import numpy as np

from .estimators import (
    EstimatedClip, FilteredOutError, HandCandidate, HandEstimator, Reconstruction,
    VideoProcessor, filter_candidates, select_non_overlapping,
)


TARGET_FPS = 30
_HAWOR_LOCK = threading.RLock()



@dataclass
class CandidateChunk:
    chunk_id: str
    hand_side: str
    frames: np.ndarray
    boxes: np.ndarray
    confidence: np.ndarray
    joints_cam: np.ndarray
    vertices_cam: np.ndarray
    parameters: dict[str, list]
    masks: np.ndarray | None = None

    @property
    def projected_area(self) -> float:
        if self.masks is not None:
            return float(np.mean(self.masks.sum(axis=(1, 2))))
        size = np.maximum(self.boxes[:, 2:4] - self.boxes[:, :2], 0)
        return float(np.mean(size[:, 0] * size[:, 1]))


@contextmanager
def _hawor_environment(hawor_root: Path):
    """Make an external HaWoR checkout importable without copying its code."""
    with _HAWOR_LOCK:
        root_path = hawor_root.resolve()
        root = str(root_path)
        for name in ("hawor", "lib", "infiller", "scripts.scripts_test_video"):
            module_file = getattr(sys.modules.get(name), "__file__", None)
            if module_file and not Path(module_file).resolve().is_relative_to(root_path):
                raise RuntimeError(
                    f"{name} is already imported from another checkout: {module_file}. "
                    "Run HaWoR preprocessing in a fresh Python process."
                )
        previous_cwd = Path.cwd()
        previous_path = sys.path[:]
        sys.path.insert(0, root)
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_path


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _normalize_and_extract(video: Path, normalized: Path, sequence_dir: Path) -> list[Path]:
    normalized.parent.mkdir(parents=True, exist_ok=True)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    image_dir = sequence_dir / "extracted_images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-an",
        "-vf", f"fps={TARGET_FPS}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(normalized),
    ])
    _run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(normalized),
        "-start_number", "0", str(image_dir / "%06d.png"),
    ])
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"ffmpeg extracted no frames from {video}")
    # The research checkout uses PNG while upstream HaWoR uses JPG. Hard links
    # let either external version consume the exact same decoded frames.
    for image in images:
        jpg_alias = image.with_suffix(".jpg")
        if not jpg_alias.exists():
            os.link(image, jpg_alias)
    return images


def _detect_all_tracklets(images: list[Path], detector_path: Path, threshold: float = 0.4):
    import torch
    from ultralytics import YOLO

    detector = YOLO(str(detector_path))
    tracks: dict[int, list[dict]] = {}
    for frame, image_path in enumerate(images):
        image = __import__("cv2").imread(str(image_path))
        with torch.no_grad():
            result = detector.track(image, conf=threshold, persist=True, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        sides = result.boxes.cls.cpu().numpy().astype(int)
        ids = None if result.boxes.id is None else result.boxes.id.cpu().numpy().astype(int)
        for detection, (box, confidence, side) in enumerate(zip(boxes, confidences, sides)):
            # Keep every candidate. Missing tracker IDs become frame-local tracks
            # rather than being collapsed into one arbitrary candidate per side.
            track_id = (
                int(ids[detection]) if ids is not None
                else 10_000_000 + frame * 1_000 + detection
            )
            tracks.setdefault(track_id, []).append({
                "frame": frame,
                "box": np.r_[box, confidence].astype(np.float32),
                "side": side,
            })
    return tracks


def _contiguous_chunks(track: list[dict]) -> Iterable[list[dict]]:
    by_frame: dict[int, dict] = {}
    for item in track:
        frame = int(item["frame"])
        if frame not in by_frame or item["box"][4] > by_frame[frame]["box"][4]:
            by_frame[frame] = item
    ordered = [by_frame[key] for key in sorted(by_frame)]
    start = 0
    for index in range(1, len(ordered) + 1):
        if index == len(ordered) or ordered[index]["frame"] != ordered[index - 1]["frame"] + 1:
            if index > start:
                yield ordered[start:index]
            start = index


def _estimate_candidates(model, tracks, images, focal: float, width: int, height: int):
    import torch
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
    from lib.vis.renderer import Renderer

    candidates: list[CandidateChunk] = []
    right_faces = None
    center = [width / 2, height / 2]
    renderer = Renderer(
        width, height, focal, "cuda", bin_size=128, max_faces_per_bin=20000
    )
    camera_rotation = torch.eye(3, device="cuda").unsqueeze(0)
    camera_translation = torch.zeros(1, 3, device="cuda")
    cameras, lights = renderer.create_camera_from_cv(camera_rotation, camera_translation)
    color = torch.tensor([0, 0, 255, 255], dtype=torch.float32, device="cuda") / 255
    for track_id, track in tracks.items():
        side_value = int(np.mean([item["side"] for item in track]) >= 0.5)
        hand_side = "right" if side_value else "left"
        for chunk_number, chunk in enumerate(_contiguous_chunks(track)):
            frames = np.asarray([item["frame"] for item in chunk], dtype=np.int64)
            boxes = np.stack([item["box"] for item in chunk]).astype(np.float32)
            result = model.inference(
                np.asarray([str(images[index]) for index in frames]), boxes,
                img_focal=focal, img_center=center, do_flip=hand_side == "left",
            )
            root_matrix = result["pred_rotmat"][None, :, 0]
            pose_matrix = result["pred_rotmat"][None, :, 1:]
            translation = result["pred_trans"][None, :, 0]
            betas = result["pred_shape"][None, :]
            root_axis = rotation_matrix_to_angle_axis(root_matrix)
            pose_axis = rotation_matrix_to_angle_axis(pose_matrix)
            if hand_side == "left":
                root_axis[..., 1:] *= -1
                pose_axis[..., 1:] *= -1
            root_matrix = angle_axis_to_rotation_matrix(root_axis)
            pose_matrix = angle_axis_to_rotation_matrix(pose_axis)
            mano = run_mano_left if hand_side == "left" else run_mano
            # The external API supplies its closed mesh topology; do not copy
            # HaWoR's wrist-cap triangles into our package.
            handedness = torch.full((1, len(frames), 1), side_value)
            output = mano(
                translation, root_axis, pose_axis, betas=betas, is_right=handedness,
            )
            vertices = output["vertices"][0].detach().cpu()
            if right_faces is None:
                topology = output
                if hand_side == "left":
                    # HaWoR's renderer uses right-hand topology for both sides.
                    # MANO_LEFT's own face ordering is different; obtain the
                    # canonical closed topology from the external right API.
                    topology = run_mano(
                        translation[:, :1], root_axis[:, :1], pose_axis[:, :1],
                        betas=betas[:, :1], is_right=torch.ones((1, 1, 1)),
                    )
                right_faces = topology["faces"][0, 0].to(device="cuda", dtype=torch.long)
            faces_tensor = right_faces if hand_side == "right" else right_faces[:, [0, 2, 1]]
            masks = []
            for vertex_frame in vertices:
                _, mask = renderer.render_multiple(
                    vertex_frame[None, None].cuda(), faces_tensor,
                    color[None], cameras, lights,
                )
                masks.append(mask)
            candidates.append(CandidateChunk(
                chunk_id=f"track-{track_id}-chunk-{chunk_number}",
                hand_side=hand_side,
                frames=frames,
                boxes=boxes[:, :4],
                confidence=boxes[:, 4],
                joints_cam=output["joints"][0].detach().cpu().numpy(),
                vertices_cam=vertices.numpy(),
                parameters={
                    "init_root_orient": root_matrix.tolist(),
                    "init_hand_pose": pose_matrix.tolist(),
                    "init_trans": translation.tolist(),
                    "init_betas": betas.tolist(),
                },
                masks=np.stack(masks),
            ))
    return candidates


def _public_candidate(item: CandidateChunk) -> HandCandidate:
    return HandCandidate(
        chunk_id=item.chunk_id, hand_side=item.hand_side, frames=item.frames,
        wrist_cam=item.joints_cam[:, 0], middle_mcp_cam=item.joints_cam[:, 9],
        confidence=item.confidence, projected_area=item.projected_area, payload=item,
    )


def _filter_candidates(candidates, clip_id, checker, sampling, min_pass_ratio):
    """Compatibility wrapper for callers of the original HaWoR runner."""
    kept, report = filter_candidates(
        [_public_candidate(item) for item in candidates], clip_id, checker,
        sampling, min_pass_ratio,
    )
    return [item.payload for item in kept], report


def _select_non_overlapping(candidates):
    return [item.payload for item in select_non_overlapping([
        _public_candidate(item) for item in candidates
    ])]


def _write_hawor_intermediates(sequence_dir: Path, selected, frame_count, height, width):
    frame_chunks = {0: [], 1: []}
    masks = np.zeros((frame_count, height, width), dtype=bool)
    for item in selected:
        hand_index = 0 if item.hand_side == "left" else 1
        frame_chunks[hand_index].append(item.frames)
        target = sequence_dir / "cam_space" / str(hand_index)
        target.mkdir(parents=True, exist_ok=True)
        name = f"{item.frames[0]}_{item.frames[-1]}.json"
        (target / name).write_text(json.dumps(item.parameters), encoding="utf-8")
        if item.masks is None:
            raise RuntimeError("candidate mesh masks are required for HaWoR SLAM")
        for frame, mask in zip(item.frames, item.masks):
            masks[frame] |= mask
    track_dir = sequence_dir / f"tracks_0_{frame_count}"
    track_dir.mkdir(parents=True, exist_ok=True)
    np.save(track_dir / "model_masks.npy", masks)
    return frame_chunks


def _load_hawor_model(checkpoint: Path, device: str):
    from scripts.scripts_test_video.hawor_video import load_hawor

    model, _ = load_hawor(str(checkpoint))
    return model.to(device).eval()


def _filling_windows(valid: np.ndarray, hand_index: int, max_length: int = 120):
    """Cover every missing frame with HaWoR-sized windows.

    A preceding valid frame is included when available to preserve temporal
    context. The simulated validity update prevents long gaps from repeatedly
    selecting the same first window.
    """
    state = np.asarray(valid, dtype=bool).copy()
    windows = []
    while np.any(~state[hand_index]):
        cursor = int(np.flatnonzero(~state[hand_index])[0])
        both_valid = np.flatnonzero(state[:, :cursor].all(axis=0))
        start = int(both_valid[-1]) if len(both_valid) else cursor
        start = max(start, cursor - max_length + 1)
        end = min(state.shape[1], start + max_length)
        if end <= cursor:
            raise RuntimeError("could not construct a filling window for missing frames")
        windows.append((start, end))
        state[:, start:end] = True
    return windows


def _network_fill(model, encoded, valid, device):
    """Adapt an upstream canonical window to the external network's API."""
    import torch

    length, features = encoded.shape
    # This checkpoint consumes 120 steps, two hands and 109 features per hand.
    padded = np.pad(encoded, ((0, 120 - length), (0, 0)), mode="edge")
    observed = np.pad(valid.all(axis=0), (0, 120 - length), constant_values=True)
    values = torch.as_tensor(padded, device=device).reshape(120, 1, features)
    observed = torch.as_tensor(observed, device=device)
    with torch.no_grad():
        result = model(
            values,
            torch.zeros((120, 120), dtype=torch.bool, device=device),
            observed.to(values.dtype).reshape(120, 1, 1),
            (~observed).reshape(1, 1, 1, 120).expand(1, 1, 120, 120),
        )
    return result[:length, 0].reshape(length, 2, 109).cpu()


def _fill_world_gaps(data, valid, *, device, model=None):
    """Use upstream canonical interpolation, optionally followed by its network.

    Only missing observations are replaced. The window coverage and absent-hand
    handling belong to our adapter; geometry and interpolation stay external.
    """
    import torch
    from lib.eval_utils.filling_utils import filling_postprocess, filling_preprocess

    present = valid.any(axis=1)
    if not present.any():
        raise ValueError("Cannot reconstruct a clip with no selected hand observations")
    if model is None:
        for hand in (0, 1):
            anchors = np.flatnonzero(valid[hand])
            if len(anchors) == 1:
                # Upstream interpolation requires two anchors. With only one,
                # constant extension is the defined adapter fallback.
                for values in data.values():
                    values[hand] = values[hand, anchors[0]].copy()
                valid[hand] = True
    if present.sum() == 1:
        source = int(present.argmax())
        # The external representation always contains both hands. Supply an
        # internal placeholder, but _world_vertices exports the absent side as 0.
        for values in data.values():
            values[1 - source] = values[source]
        valid[1 - source] = valid[source]
    for hand in (1, 0):
        for start, end in _filling_windows(valid, hand):
            observed = valid[:, start:end].copy()
            window = {name: values[:, start:end].copy() for name, values in data.items()}
            encoded, transform = filling_preprocess(dict(window, valid=observed))
            if model is None:
                predicted = torch.from_numpy(encoded).reshape(end - start, 2, 109)
            else:
                predicted = _network_fill(model, encoded, observed, device)
            filled = filling_postprocess(predicted, transform)
            for name, values in data.items():
                values[:, start:end][~observed] = filled[name][~observed]
            valid[:, start:end] = True


def _run_external_infiller(args, frame_chunks, frame_count, device: str, fill_mode: str):
    """Translate selected observations through external camera/filling APIs."""
    import torch
    from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    sequence_dir = Path(args.video_path).with_suffix("")
    slam_path = sequence_dir / "SLAM" / f"hawor_slam_w_scale_0_{frame_count}.npz"
    _, _, rotation, translation = load_slam_cam(str(slam_path))
    fields = {
        "trans": ("init_trans", 3), "rot": ("init_root_orient", 3),
        "hand_pose": ("init_hand_pose", 45), "betas": ("init_betas", 10),
    }
    data = {name: np.zeros((2, frame_count, size), np.float32)
            for name, (_, size) in fields.items()}
    valid = np.zeros((2, frame_count), dtype=bool)
    for hand, side in enumerate(("left", "right")):
        for frames in frame_chunks[hand]:
            path = sequence_dir / "cam_space" / str(hand) / f"{frames[0]}_{frames[-1]}.json"
            parameters = {key: torch.tensor(value) for key, value in
                          json.loads(path.read_text(encoding="utf-8")).items()}
            world = cam2world_convert(rotation[frames], translation[frames], parameters, side)
            for name, (key, size) in fields.items():
                value = world[key]
                if name == "hand_pose" and value.shape[-2:] == (3, 3):
                    value = rotation_matrix_to_angle_axis(value)
                data[name][hand, frames] = value.detach().cpu().numpy().reshape(len(frames), size)
            valid[hand, frames] = True

    model = None
    if fill_mode == "network":
        from infiller.lib.model.network import TransformerModel

        # Architecture arguments describe the external pretrained checkpoint.
        model = TransformerModel(
            seq_len=120, input_dim=218, d_model=384, nhead=8, d_hid=2048,
            nlayers=8, dropout=0.05, out_dim=218, masked_attention_stage=True,
        ).to(device)
        checkpoint = torch.load(args.infiller_weight, map_location=device)
        model.load_state_dict(checkpoint["transformer_encoder_state_dict"])
        model.eval()
    _fill_world_gaps(data, valid, device=device, model=model)
    return tuple(torch.from_numpy(data[name]) for name in fields) + (valid,)


def _normalize_slam_output(sequence_dir: Path, frame_count: int):
    """Align external camera-file dtypes before using its geometry helpers.

    Stock HaWoR can save a float64 metric scale beside a float32 trajectory.
    Its loader then mixes Float/Double tensors in einsum on newer NumPy/PyTorch.
    Normalize only our generated output, leaving the external source untouched.
    """
    path = sequence_dir / "SLAM" / f"hawor_slam_w_scale_0_{frame_count}.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = dict(archive)
    values["traj"] = values["traj"].astype(np.float32)
    values["scale"] = values["scale"].astype(np.float32)
    np.savez(path, **values)


def _camera_matrices(sequence_dir: Path, frame_count: int):
    from lib.eval_utils.custom_utils import load_slam_cam

    slam = sequence_dir / "SLAM" / f"hawor_slam_w_scale_0_{frame_count}.npz"
    rotation, translation, _, _ = load_slam_cam(str(slam))
    rotation = np.asarray(rotation.detach().cpu() if hasattr(rotation, "detach") else rotation)
    translation = np.asarray(
        translation.detach().cpu() if hasattr(translation, "detach") else translation
    )
    if len(rotation) != frame_count:
        raise RuntimeError(f"SLAM returned {len(rotation)} poses for {frame_count} frames")
    matrices = np.broadcast_to(np.eye(4, dtype=np.float32), (frame_count, 4, 4)).copy()
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = translation
    return matrices


def _world_vertices(world_data, selected_sides: set[str]):
    from hawor.utils.process import run_mano, run_mano_left

    trans, root, pose, betas, _ = world_data
    frame_count = int(trans.shape[1])
    result = np.zeros((frame_count, 1556, 3), dtype=np.float32)
    for index, (side, mano, offset) in enumerate((
        ("left", run_mano_left, 0), ("right", run_mano, 778),
    )):
        if side not in selected_sides:
            continue
        output = mano(
            trans[index:index + 1], root[index:index + 1], pose[index:index + 1],
            betas=betas[index:index + 1],
        )
        vertices = output["vertices"][0].detach().cpu().numpy()
        result[:, offset:offset + 778] = vertices[:, :778]
    return result


def _run_hawor_slam(args, frame_count, threshold):
    with _HAWOR_LOCK:
        from scripts.scripts_test_video import hawor_slam as upstream

        # Research checkouts expose this option directly. Stock HaWoR calls the
        # low-level API without forwarding it; bind the option for this call only.
        # Neither path changes files in the external installation.
        if "thres" in inspect.signature(upstream.hawor_slam).parameters:
            return upstream.hawor_slam(args, 0, frame_count, thres=threshold)
        original = upstream.run_slam

        def configured_slam(*args, **kwargs):
            kwargs["filter_thresh"] = threshold
            return original(*args, **kwargs)

        upstream.run_slam = configured_slam
        try:
            return upstream.hawor_slam(args, 0, frame_count)
        finally:
            upstream.run_slam = original


class HaWoREstimator(HandEstimator):
    """External HaWoR model behind the common estimate/reconstruct interface.

    Detection and hand prediction expose every candidate. SLAM and filling run
    only after the shared protagonist filter has selected the retained hands.
    """

    def __init__(
        self, *, hawor_root, device="cuda", checkpoint=None, detector=None,
        infiller_weight=None, img_focal=None, fill_mode="interpolation",
        slam_filter_threshold=1.0, min_num_frames=81,
    ):
        import torch

        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("HaWoR preprocessing requires --device cuda or cuda:N")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; HaWoR DROID-SLAM requires a CUDA GPU")
        torch.cuda.set_device(0 if target.index is None else target.index)
        self.device = device
        self.root = Path(hawor_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint or self.root / "weights/hawor/checkpoints/hawor.ckpt").resolve()
        self.detector = Path(detector or self.root / "weights/external/detector.pt").resolve()
        self.infiller = Path(infiller_weight or self.root / "weights/hawor/checkpoints/infiller.pt").resolve()
        self.metric = self.root / "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"
        self.focal = float(img_focal if img_focal is not None else 600.0)
        if not np.isfinite(self.focal) or self.focal <= 0:
            raise ValueError("img_focal must be finite and positive")
        if fill_mode not in {"interpolation", "network"}:
            raise ValueError("fill_mode must be interpolation or network")
        self.fill_mode = fill_mode
        self.slam_filter_threshold = slam_filter_threshold
        self.min_num_frames = min_num_frames
        required = [self.root, self.checkpoint, self.detector, self.metric]
        if fill_mode == "network":
            required.append(self.infiller)
        for path in required:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing external HaWoR resource: {path}. Install upstream HaWoR "
                    "and its model assets; no handsonworld_adapter.py is required."
                )
        self.provenance = {
            "hawor_root": str(self.root), "checkpoint": str(self.checkpoint),
            "detector": str(self.detector), "infiller_weight": str(self.infiller),
            "metric_depth_weight": str(self.metric), "img_focal": self.focal,
            "fill_mode": self.fill_mode, "slam_filter_threshold": self.slam_filter_threshold,
            "target_fps": TARGET_FPS, "min_num_frames": self.min_num_frames,
        }
        # Stock HaWoR selects the multiprocessing spawn context at SLAM import.
        # Import it before detector/model/body-fitter construction can select a
        # different default context; this imports dependencies, not SLAM work.
        with _hawor_environment(self.root):
            from scripts.scripts_test_video import hawor_slam  # noqa: F401
        self.model = None

    def estimate(self, video: Path, work_dir: Path) -> EstimatedClip:
        video = video.expanduser().resolve()
        work_dir = work_dir.resolve()
        # Keep HaWoR's <video parent>/<video stem> convention inside this clip's
        # workspace; model resources still resolve relative to the checkout.
        normalized = work_dir / "video.mp4"
        sequence_dir = normalized.with_suffix("")
        with _hawor_environment(self.root):
            print(f"[{video.stem}] normalize and extract {TARGET_FPS} FPS frames")
            images = _normalize_and_extract(video, normalized, sequence_dir)
            if len(images) < self.min_num_frames:
                raise FilteredOutError(
                    f"clip has {len(images)} frames; at least {self.min_num_frames} required"
                )
            import cv2
            first = cv2.imread(str(images[0]))
            height, width = first.shape[:2]
            if self.model is None:
                self.model = _load_hawor_model(self.checkpoint, self.device)
            tracks = _detect_all_tracklets(images, self.detector)
            print(f"[{video.stem}] detected {len(tracks)} raw tracklets")
            candidates = _estimate_candidates(
                self.model, tracks, images, self.focal, width, height,
            )
        return EstimatedClip(
            video=normalized, num_frames=len(images), width=width, height=height,
            intrinsics=np.asarray([
                [self.focal, 0, width / 2], [0, self.focal, height / 2], [0, 0, 1],
            ], dtype=np.float32),
            candidates=[_public_candidate(item) for item in candidates],
        )

    def reconstruct(self, clip: EstimatedClip, selected: list[HandCandidate]) -> Reconstruction:
        sequence_dir = clip.video.with_suffix("")
        candidates = [item.payload for item in selected]
        with _hawor_environment(self.root):
            print(f"Selected {len(candidates)} chunks; running camera SLAM")
            frame_chunks = _write_hawor_intermediates(
                sequence_dir, candidates, clip.num_frames, clip.height, clip.width,
            )
            external_args = argparse.Namespace(
                video_path=str(clip.video), img_focal=self.focal,
                infiller_weight=str(self.infiller), first_person_view_check="none",
            )
            _run_hawor_slam(external_args, clip.num_frames, self.slam_filter_threshold)
            _normalize_slam_output(sequence_dir, clip.num_frames)
            print("Camera SLAM complete; filling missing hand frames")
            world_data = _run_external_infiller(
                external_args, frame_chunks, clip.num_frames, self.device, self.fill_mode,
            )
            return Reconstruction(
                camera=_camera_matrices(sequence_dir, clip.num_frames),
                hand_vertices=_world_vertices(world_data, {item.hand_side for item in selected}),
            )


class HaWoRProcessor(VideoProcessor):
    """Compatibility entry point; filtering/export now use the common processor."""

    def __init__(self, args: argparse.Namespace):
        from .video_preprocessing import make_checker

        estimator = HaWoREstimator(
            hawor_root=args.hawor_root, device=args.device,
            checkpoint=args.checkpoint, detector=args.detector,
            infiller_weight=args.infiller_weight, img_focal=args.img_focal,
            fill_mode=args.fill_mode, slam_filter_threshold=args.slam_filter_threshold,
            min_num_frames=args.min_num_frames,
        )
        super().__init__(
            estimator, make_checker(args), sampling=args.sampling,
            min_pass_ratio=args.min_pass_ratio, min_num_frames=args.min_num_frames,
        )


# Retain the original runner's Python and CLI entry points.
from .video_preprocessing import build_parser, main, run_manifest  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
