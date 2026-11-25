"""Estimator-neutral video preparation and protagonist-hand filtering.

Estimators produce lightweight camera-space hand candidates first.  The
shared processor filters those candidates before asking the estimator to run
its potentially expensive camera and dense-hand reconstruction stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ego_hand_filter.decision import filter_chunks
from ego_hand_filter.schema import HandObservations

from .videos import probe_video


Checker = Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray, dict]]


class FilteredOutError(RuntimeError):
    """Expected dataset rejection rather than a processing failure."""


@dataclass
class HandCandidate:
    """One contiguous hand candidate in camera-space metres."""

    chunk_id: str
    hand_side: str
    frames: np.ndarray
    wrist_cam: np.ndarray
    middle_mcp_cam: np.ndarray
    confidence: np.ndarray
    projected_area: float
    valid: np.ndarray | None = None
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if self.hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be 'left' or 'right'")

        frames = np.asarray(self.frames)
        if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu":
            raise ValueError("frames must be a non-empty one-dimensional integer array")
        frames = frames.astype(np.int64, copy=False)
        if np.any(frames < 0) or np.any(np.diff(frames) <= 0):
            raise ValueError("frames must be sorted, unique, and non-negative")

        size = len(frames)
        wrist = np.asarray(self.wrist_cam, dtype=np.float32)
        middle = np.asarray(self.middle_mcp_cam, dtype=np.float32)
        if wrist.shape != (size, 3):
            raise ValueError(f"wrist_cam must have shape ({size}, 3), got {wrist.shape}")
        if middle.shape != (size, 3):
            raise ValueError(
                f"middle_mcp_cam must have shape ({size}, 3), got {middle.shape}"
            )
        finite = np.isfinite(wrist).all(axis=1) & np.isfinite(middle).all(axis=1)
        if self.valid is None:
            valid = finite
        else:
            valid = np.asarray(self.valid, dtype=bool)
            if valid.shape != (size,):
                raise ValueError(f"valid must have shape ({size},), got {valid.shape}")
            if np.any(valid & ~finite):
                raise ValueError("valid observations must contain finite camera-space anchors")

        confidence = np.asarray(self.confidence, dtype=np.float32)
        if confidence.shape != (size,) or not np.isfinite(confidence).all():
            raise ValueError(f"confidence must be finite with shape ({size},)")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence values must be between 0 and 1")
        area = float(self.projected_area)
        if not math.isfinite(area) or area < 0:
            raise ValueError("projected_area must be finite and non-negative")

        self.frames = frames
        self.wrist_cam = wrist
        self.middle_mcp_cam = middle
        self.confidence = confidence
        self.projected_area = area
        self.valid = valid


@dataclass
class EstimatedClip:
    """Estimator output required by the shared filtering stage."""

    video: Path
    num_frames: int
    width: int
    height: int
    intrinsics: np.ndarray
    candidates: list[HandCandidate]
    context: Any = None

    def __post_init__(self) -> None:
        self.video = Path(self.video)
        for name in ("num_frames", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        self.intrinsics = validate_intrinsics(self.intrinsics)
        self.candidates = list(self.candidates)
        if not all(isinstance(item, HandCandidate) for item in self.candidates):
            raise TypeError("candidates must contain only HandCandidate objects")
        keys = [(item.chunk_id, item.hand_side) for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate (chunk_id, hand_side) pairs must be unique per clip")
        for item in self.candidates:
            if int(item.frames[-1]) >= self.num_frames:
                raise ValueError(
                    f"candidate {item.chunk_id}/{item.hand_side} contains frame "
                    f"{int(item.frames[-1])} outside clip length {self.num_frames}"
                )


@dataclass
class Reconstruction:
    camera: np.ndarray
    hand_vertices: np.ndarray


class HandEstimator(ABC):
    """Interface implemented by HaWoR and alternative hand estimators."""

    @abstractmethod
    def estimate(self, video: Path, work_dir: Path) -> EstimatedClip:
        """Return sparse candidates and any private reconstruction context."""

    @abstractmethod
    def reconstruct(
        self, clip: EstimatedClip, selected: list[HandCandidate]
    ) -> Reconstruction:
        """Reconstruct camera poses and dense world-space hand vertices."""


def _validate_filter_settings(sampling: str, min_pass_ratio: float) -> None:
    if sampling not in {"chunk_first", "all_frames"}:
        raise ValueError("sampling must be 'chunk_first' or 'all_frames'")
    if not math.isfinite(min_pass_ratio) or not 0 <= min_pass_ratio <= 1:
        raise ValueError("min_pass_ratio must be finite and between 0 and 1")


def validate_intrinsics(value: np.ndarray) -> np.ndarray:
    """Return a validated zero-skew pinhole intrinsic matrix."""

    intrinsics = np.asarray(value, dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be a finite array with shape (3, 3)")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("intrinsics must have positive focal lengths")
    if not np.allclose((intrinsics[0, 1], intrinsics[1, 0]), (0, 0), atol=1e-6):
        raise ValueError("intrinsics must use the zero-skew pinhole form")
    if not np.allclose(intrinsics[2], (0, 0, 1), atol=1e-6):
        raise ValueError("intrinsics must use the homogeneous bottom row [0, 0, 1]")
    return intrinsics


def filter_candidates(
    candidates: list[HandCandidate],
    clip_id: str,
    checker: Checker,
    sampling: str,
    min_pass_ratio: float,
) -> tuple[list[HandCandidate], list[dict]]:
    """Apply the public body-based filter while preserving side-specific keys."""

    _validate_filter_settings(sampling, min_pass_ratio)
    if not candidates:
        return [], []
    clip_ids: list[str] = []
    chunk_ids: list[str] = []
    sides: list[str] = []
    frames: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    middles: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    for item in candidates:
        size = len(item.frames)
        clip_ids.extend([clip_id] * size)
        chunk_ids.extend([item.chunk_id] * size)
        sides.extend([item.hand_side] * size)
        frames.append(item.frames)
        wrists.append(item.wrist_cam)
        middles.append(item.middle_mcp_cam)
        valid.append(np.asarray(item.valid, dtype=bool))
        confidence.append(item.confidence)
        areas.append(np.full(size, item.projected_area, dtype=np.float32))
    observations = HandObservations(
        clip_id=np.asarray(clip_ids),
        chunk_id=np.asarray(chunk_ids),
        frame_index=np.concatenate(frames),
        hand_side=np.asarray(sides),
        wrist_cam=np.concatenate(wrists),
        middle_mcp_cam=np.concatenate(middles),
        valid=np.concatenate(valid),
        confidence=np.concatenate(confidence),
        projected_area=np.concatenate(areas),
    )
    decisions = filter_chunks(
        observations,
        checker,
        sampling=sampling,
        min_pass_ratio=min_pass_ratio,
    )
    kept_keys = {
        (decision.chunk_id, decision.hand_side)
        for decision in decisions
        if decision.keep
    }
    kept = [item for item in candidates if (item.chunk_id, item.hand_side) in kept_keys]
    return kept, [decision.to_dict() for decision in decisions]


def select_non_overlapping(candidates: list[HandCandidate]) -> list[HandCandidate]:
    """Keep area-ranked, whole candidates whose frames do not overlap per side."""

    selected: list[HandCandidate] = []
    for side in ("left", "right"):
        occupied: set[int] = set()
        ranked = sorted(
            (item for item in candidates if item.hand_side == side),
            key=lambda item: item.projected_area,
            reverse=True,
        )
        for item in ranked:
            item_frames = set(item.frames.tolist())
            if item_frames.isdisjoint(occupied):
                selected.append(item)
                occupied.update(item_frames)
    return sorted(selected, key=lambda item: (item.hand_side, int(item.frames[0])))


def validate_reconstruction(value: Reconstruction, num_frames: int) -> Reconstruction:
    """Return validated camera and dense world-space hand arrays."""

    if not isinstance(value, Reconstruction):
        raise TypeError("estimator reconstruct() must return Reconstruction")
    camera = np.asarray(value.camera, dtype=np.float32)
    hands = np.asarray(value.hand_vertices, dtype=np.float32)
    if camera.shape != (num_frames, 4, 4):
        raise ValueError(
            f"camera must have shape ({num_frames}, 4, 4), got {camera.shape}"
        )
    if hands.shape != (num_frames, 1556, 3):
        raise ValueError(
            f"hand_vertices must have shape ({num_frames}, 1556, 3), got {hands.shape}"
        )
    if not np.isfinite(camera).all() or not np.isfinite(hands).all():
        raise ValueError("reconstruction arrays must contain only finite values")
    if not np.allclose(camera[:, 3], (0, 0, 0, 1), atol=1e-5):
        raise ValueError("camera matrices must use homogeneous [0, 0, 0, 1] bottom rows")
    rotations = camera[:, :3, :3]
    identity = np.broadcast_to(np.eye(3, dtype=np.float32), rotations.shape)
    if not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), identity, atol=1e-3):
        raise ValueError("camera rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3):
        raise ValueError("camera rotations must be proper rotations with determinant 1")
    return Reconstruction(camera=camera, hand_vertices=hands)


class VideoProcessor:
    """Shared estimator -> filter -> reconstruction -> manifest pipeline."""

    def __init__(
        self,
        estimator: HandEstimator,
        checker: Checker,
        *,
        sampling: str = "chunk_first",
        min_pass_ratio: float = 0.9,
        min_num_frames: int = 81,
        provenance: dict | None = None,
    ) -> None:
        _validate_filter_settings(sampling, min_pass_ratio)
        if isinstance(min_num_frames, bool) or not isinstance(
            min_num_frames, (int, np.integer)
        ) or min_num_frames <= 0:
            raise ValueError("min_num_frames must be a positive integer")
        self.estimator = estimator
        self.checker = checker
        self.sampling = sampling
        self.min_pass_ratio = float(min_pass_ratio)
        self.min_num_frames = int(min_num_frames)
        self.provenance = dict(provenance or {})

    def __call__(self, video: Path, prompt: str, output_dir: Path) -> dict:
        video = Path(video).expanduser().resolve()
        output_dir = Path(output_dir)
        digest = hashlib.sha1(str(video).encode()).hexdigest()[:10]
        slug = f"{video.stem}-{digest}"
        work_dir = output_dir / "artifacts" / slug
        work_dir.mkdir(parents=True, exist_ok=True)

        clip = self.estimator.estimate(video, work_dir)
        if not isinstance(clip, EstimatedClip):
            raise TypeError("estimator estimate() must return EstimatedClip")
        # Estimator objects are intentionally extensible and mutable. Snapshot
        # every validated public field before calling backend reconstruction so
        # an accidental mutation inside that callback cannot alter the output
        # contract after validation.
        clip_video = clip.video.resolve()
        num_frames = clip.num_frames
        width = clip.width
        height = clip.height
        intrinsics = clip.intrinsics.copy()
        decoded = probe_video(clip_video)
        expected = (num_frames, width, height)
        if decoded != expected:
            raise ValueError(
                "estimated video metadata mismatch: "
                f"declared frames/width/height={expected}, decoded={decoded}"
            )
        if num_frames < self.min_num_frames:
            raise FilteredOutError(
                f"clip has {num_frames} frames; at least {self.min_num_frames} required"
            )

        print(
            f"[{slug}] filtering {len(clip.candidates)} candidate chunks "
            f"with sampling={self.sampling}",
            flush=True,
        )
        filtered, decisions = filter_candidates(
            clip.candidates,
            slug,
            self.checker,
            self.sampling,
            self.min_pass_ratio,
        )
        (work_dir / "filter_decisions.json").write_text(
            json.dumps(decisions, indent=2), encoding="utf-8"
        )
        provenance = {
            **self.provenance,
            "estimator": f"{type(self.estimator).__module__}.{type(self.estimator).__qualname__}",
            "sampling": self.sampling,
            "min_pass_ratio": self.min_pass_ratio,
            "min_num_frames": self.min_num_frames,
        }
        (work_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        selected = select_non_overlapping(filtered)
        if not selected:
            raise FilteredOutError("no protagonist-hand candidate passed filtering")
        reconstruction = validate_reconstruction(
            self.estimator.reconstruct(clip, selected), num_frames
        )
        camera_path = work_dir / "cam_pose.npy"
        hands_path = work_dir / "hand_pose.npy"
        intrinsics_path = work_dir / "camera_intrinsics.npy"
        np.save(camera_path, reconstruction.camera)
        np.save(hands_path, reconstruction.hand_vertices)
        np.save(intrinsics_path, intrinsics)
        return {
            "input_video": str(video),
            "video": str(clip_video),
            "prompt": prompt,
            "cam_pose": str(camera_path.resolve()),
            "hand_pose": str(hands_path.resolve()),
            "camera_intrinsics": str(intrinsics_path.resolve()),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "status": "success",
            "message": "",
        }
