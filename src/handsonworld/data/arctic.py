"""Convert the official ARCTIC egocentric release into training controls."""

from __future__ import annotations

import json
import math
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ArcticSequence:
    subject: str
    sequence: str
    split: str
    mano_path: Path


def resolve_data_root(base_dir: Path) -> Path:
    """Resolve either the official ``unpack`` root or ``arctic_data/data``."""
    base_dir = base_dir.expanduser().resolve()
    candidates = (
        base_dir,
        base_dir / "arctic_data" / "data",
        base_dir / "data",
    )
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("cropped_images", "raw_seqs", "splits_json", "meta")):
            return candidate
    raise FileNotFoundError(
        f"Could not find an ARCTIC data root under {base_dir}. Expected "
        "cropped_images/, raw_seqs/, splits_json/, and meta/."
    )


def resolve_mano_dir(base_dir: Path, data_root: Path, mano_dir: Path | None = None) -> Path:
    """Locate MANO files from an explicit path or the official unpack layout."""
    if mano_dir is not None:
        explicit = mano_dir.expanduser().resolve()
        if all((explicit / name).is_file() for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")):
            return explicit
        raise FileNotFoundError(
            f"Explicit MANO directory {explicit} does not contain MANO_LEFT.pkl and MANO_RIGHT.pkl."
        )
    candidates = [
        base_dir.expanduser().resolve() / "body_models" / "mano",
        data_root.parents[2] / "body_models" / "mano",
        data_root.parents[1] / "body_models" / "mano",
        data_root.parent / "body_models" / "mano",
    ]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")):
            return candidate
    raise FileNotFoundError(
        "Could not find MANO_LEFT.pkl and MANO_RIGHT.pkl. Pass --mano-dir or "
        "place them under <ARCTIC unpack>/body_models/mano/."
    )


def jpeg_size(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions without requiring Pillow or OpenCV."""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG file: {path}")
        while True:
            byte = handle.read(1)
            while byte and byte != b"\xff":
                byte = handle.read(1)
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                raise ValueError(f"Could not find JPEG dimensions: {path}")
            code = marker[0]
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                raise ValueError(f"Truncated JPEG marker: {path}")
            length = int.from_bytes(length_bytes, "big")
            if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                payload = handle.read(length - 2)
                return int.from_bytes(payload[3:5], "big"), int.from_bytes(payload[1:3], "big")
            handle.seek(length - 2, os.SEEK_CUR)


def center_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Center-crop ARCTIC ego images to 4:3 (840x600 -> 800x600)."""
    scale = min(width // 4, height // 3)
    if scale < 1:
        raise ValueError(f"Image is too small for a 4:3 crop: {width}x{height}")
    crop_width, crop_height = 4 * scale, 3 * scale
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def processed_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    image_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    image_width, image_height = image_size
    output = np.asarray(intrinsics, dtype=np.float32).copy()
    output[0] *= image_width / source_width
    output[1] *= image_height / source_height
    output[0, 2] -= crop_box[0]
    output[1, 2] -= crop_box[1]
    return output


def sequence_prompt(sequence: str) -> str:
    parts = sequence.split("_")
    obj = parts[0].replace("capsulemachine", "capsule machine").replace(
        "espressomachine", "espresso machine"
    ).replace("waffleiron", "waffle iron")
    action = parts[1] if len(parts) > 1 else "interacting"
    phrase = f"grabbing a {obj}" if action == "grab" else f"using a {obj}" if action == "use" else f"{action} with a {obj}"
    return f"First-person egocentric hands {phrase}."


def selected_sequences(data_root: Path, protocol: str, split: str) -> list[ArcticSequence]:
    split_path = data_root / "splits_json" / f"protocol_{protocol}.json"
    split_map = json.loads(split_path.read_text(encoding="utf-8"))
    if split not in split_map:
        raise ValueError(f"Split {split!r} is not present in {split_path}")
    result = []
    for relative in split_map[split]:
        subject, sequence = relative.split("/", 1)
        result.append(ArcticSequence(subject, sequence, split, data_root / "raw_seqs" / subject / f"{sequence}.mano.npy"))
    return result


def load_world_to_camera(mano_path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(mano_path).replace(".mano.npy", ".egocam.dist.npy"))
    data = np.load(path, allow_pickle=True).item()
    rotation = np.asarray(data["R_k_cam_np"], dtype=np.float32)
    translation = np.asarray(data["T_k_cam_np"], dtype=np.float32).reshape(-1, 3)
    world_to_camera = np.repeat(np.eye(4, dtype=np.float32)[None], len(rotation), axis=0)
    world_to_camera[:, :3, :3] = rotation
    world_to_camera[:, :3, 3] = translation
    return world_to_camera, np.asarray(data["intrinsics"], dtype=np.float32)


def image_frames(data_root: Path, sequence: ArcticSequence, offset: int, pose_frames: int) -> tuple[list[Path], np.ndarray]:
    directory = data_root / "cropped_images" / sequence.subject / sequence.sequence / "0"
    pairs = []
    for path in sorted(directory.glob("*.jpg")):
        try:
            index = int(path.stem) - offset
        except ValueError:
            continue
        if 0 <= index < pose_frames:
            pairs.append((path, index))
    return [item[0] for item in pairs], np.asarray([item[1] for item in pairs], dtype=np.int64)


def select_frames(paths: list[Path], indices: np.ndarray, skip: int, stride: int) -> tuple[list[Path], np.ndarray]:
    return paths[skip::stride], indices[skip::stride]


def load_mano_layers(mano_dir: Path, device: str):
    from smplx import MANOLayer

    common = {"model_path": str(mano_dir), "use_pca": False, "flat_hand_mean": False}
    return MANOLayer(**common, is_rhand=False).to(device), MANOLayer(**common, is_rhand=True).to(device)


def _run_mano(layer, side: dict, indices: np.ndarray, device: str):
    import torch
    from smplx.lbs import batch_rodrigues

    rot = torch.from_numpy(np.asarray(side["rot"], dtype=np.float32)[indices]).to(device)
    pose = torch.from_numpy(np.asarray(side["pose"], dtype=np.float32)[indices]).to(device)
    trans = torch.from_numpy(np.asarray(side["trans"], dtype=np.float32)[indices]).to(device)
    shape = np.asarray(side["shape"], dtype=np.float32).reshape(1, 10)
    shape = torch.from_numpy(np.repeat(shape, len(indices), axis=0)).to(device)
    full_pose = torch.cat((rot, pose), dim=1) + layer.pose_mean.to(device=device, dtype=rot.dtype)
    matrices = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    return layer(global_orient=matrices[:, :1], hand_pose=matrices[:, 1:], betas=shape, transl=trans).vertices


def export_hand_vertices(
    mano_path: Path,
    indices: np.ndarray,
    world_to_camera: np.ndarray,
    mano_layers,
    device: str,
) -> np.ndarray:
    """Return both hands in the first ego-camera reference coordinate frame."""
    import torch

    data = np.load(mano_path, allow_pickle=True).item()
    with torch.no_grad():
        left = _run_mano(mano_layers[0], data["left"], indices, device).float()
        right = _run_mano(mano_layers[1], data["right"], indices, device).float()
        reference = torch.from_numpy(world_to_camera[0]).to(device=device, dtype=torch.float32)
        vertices = torch.cat((left, right), dim=1)
        vertices = torch.einsum("ij,tvj->tvi", reference[:3, :3], vertices) + reference[:3, 3]
    return vertices.cpu().numpy().astype(np.float32)


def relative_world_to_camera(world_to_camera: np.ndarray) -> np.ndarray:
    """Express each selected camera relative to the first selected camera."""
    return (world_to_camera @ np.linalg.inv(world_to_camera[0])[None]).astype(np.float32)


def export_video(paths: list[Path], output: Path, crop_box: tuple[int, int, int, int], fps: float) -> None:
    import imageio.v2 as imageio

    left, top, right, bottom = crop_box
    frames = [imageio.imread(path)[top:bottom, left:right, :3] for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def load_misc(data_root: Path) -> dict:
    return json.loads((data_root / "meta" / "misc.json").read_text(encoding="utf-8"))


def iter_plans(
    data_root: Path,
    protocol: str,
    split: str,
    skip: int,
    stride: int,
    issues: list[dict[str, str]] | None = None,
) -> Iterator[tuple[ArcticSequence, list[Path], np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int, int, int]]]:
    misc = load_misc(data_root)
    for sequence in selected_sequences(data_root, protocol, split):
        try:
            if not sequence.mano_path.is_file():
                raise FileNotFoundError(sequence.mano_path)
            world_to_camera, intrinsics = load_world_to_camera(sequence.mano_path)
            subject_meta = misc[sequence.subject]
            paths, indices = image_frames(data_root, sequence, int(subject_meta.get("ioi_offset", 0)), len(world_to_camera))
            paths, indices = select_frames(paths, indices, skip, stride)
            if not paths:
                raise FileNotFoundError(
                    data_root / "cropped_images" / sequence.subject / sequence.sequence / "0"
                )
            image_size = jpeg_size(paths[0])
            source_size = tuple(subject_meta.get("image_size", [image_size])[0])
            crop_box = center_crop_box(*image_size)
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            if issues is None:
                raise
            issues.append({
                "subject": sequence.subject,
                "sequence": sequence.sequence,
                "split": sequence.split,
                "status": "missing",
                "message": str(exc),
            })
            continue
        yield sequence, paths, indices, world_to_camera, intrinsics, source_size, image_size, crop_box


def sequence_settings(
    sequence: ArcticSequence,
    indices: np.ndarray,
    image_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    protocol: str,
    skip: int,
    stride: int,
    fps: float,
) -> dict:
    indices = np.asarray(indices, dtype=np.int64)
    return {
        "format": "handsonworld-arctic-v2",
        "source_mano": str(sequence.mano_path.resolve()),
        "subject": sequence.subject,
        "sequence": sequence.sequence,
        "protocol": protocol,
        "split": sequence.split,
        "skip_initial_frames": skip,
        "frame_stride": stride,
        "source_fps": fps,
        "output_fps": fps / stride,
        "num_frames": int(len(indices)),
        "first_pose_index": int(indices[0]),
        "last_pose_index": int(indices[-1]),
        "pose_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "image_size": list(image_size),
        "crop_box": list(crop_box),
    }


def validate_existing_outputs(
    video_path: Path,
    hand_path: Path,
    camera_path: Path,
    intrinsics_path: Path,
    settings_path: Path,
    expected: dict,
) -> None:
    """Reject stale or incomplete sequence outputs before a resume."""
    required = (video_path, hand_path, camera_path, intrinsics_path, settings_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "Incomplete existing ARCTIC output; rerun with --overwrite. Missing: "
            + ", ".join(missing)
        )
    actual = json.loads(settings_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(
            f"Existing settings differ for {expected['subject']}/{expected['sequence']}; "
            "rerun with --overwrite to regenerate aligned files."
        )
    frame_count = expected["num_frames"]
    arrays = (
        (hand_path, (frame_count, 1556, 3)),
        (camera_path, (frame_count, 4, 4)),
        (intrinsics_path, (3, 3)),
    )
    for path, shape in arrays:
        value = np.load(path, mmap_mode="r")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(
                f"Invalid existing {path}: expected finite {shape}, found {value.shape}; rerun with --overwrite."
            )
    import imageio.v3 as iio
    import imageio_ffmpeg

    metadata = iio.immeta(video_path)
    video_frames, _seconds = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    crop = expected["crop_box"]
    expected_size = (crop[2] - crop[0], crop[3] - crop[1])
    actual_size = tuple(metadata.get("size", ()))
    actual_fps = float(metadata.get("fps", math.nan))
    if video_frames != frame_count or actual_size != expected_size or not math.isclose(
        actual_fps, expected["output_fps"], rel_tol=0, abs_tol=1e-3
    ):
        raise ValueError(
            f"Invalid existing {video_path}: frames={video_frames}, size={actual_size}, fps={actual_fps}; "
            f"expected frames={frame_count}, size={expected_size}, fps={expected['output_fps']}. "
            "Rerun with --overwrite."
        )
