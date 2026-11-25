"""Versioned, estimator-neutral hand observation schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = "ego-hand-observations-v1"
CAMERA_FRAME = "camera"
METERS = "meters"


def _string_vector(value: object, name: str, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != size:
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if array.dtype.kind not in "USO":
        raise ValueError(f"{name} must contain strings")
    result = array.astype(str)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{name} must not contain empty strings")
    return result


def _scalar_string(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


@dataclass(frozen=True)
class HandObservations:
    """One row per hand observation.

    Rows are grouped by ``(clip_id, chunk_id, hand_side)``. Coordinates are
    required to be camera-space metres so a body fitter never has to guess a
    convention or scale.
    """

    clip_id: np.ndarray
    chunk_id: np.ndarray
    frame_index: np.ndarray
    hand_side: np.ndarray
    wrist_cam: np.ndarray
    middle_mcp_cam: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    projected_area: np.ndarray | None = None
    schema_version: str = SCHEMA_VERSION
    coordinate_frame: str = CAMERA_FRAME
    units: str = METERS

    def __post_init__(self) -> None:
        wrist = np.asarray(self.wrist_cam, dtype=np.float32)
        middle = np.asarray(self.middle_mcp_cam, dtype=np.float32)
        if wrist.ndim != 2 or wrist.shape[1:] != (3,):
            raise ValueError(f"wrist_cam must have shape (N, 3), got {wrist.shape}")
        size = len(wrist)
        if middle.shape != (size, 3):
            raise ValueError(f"middle_mcp_cam must have shape ({size}, 3), got {middle.shape}")

        clip_id = _string_vector(self.clip_id, "clip_id", size)
        chunk_id = _string_vector(self.chunk_id, "chunk_id", size)
        hand_side = _string_vector(self.hand_side, "hand_side", size)
        if not np.all(np.isin(hand_side, ("left", "right"))):
            raise ValueError("hand_side values must be 'left' or 'right'")

        frame_index = np.asarray(self.frame_index)
        if frame_index.shape != (size,) or frame_index.dtype.kind not in "iu":
            raise ValueError(f"frame_index must be an integer array with shape ({size},)")
        valid = np.asarray(self.valid, dtype=bool)
        if valid.shape != (size,):
            raise ValueError(f"valid must have shape ({size},)")
        confidence = np.asarray(self.confidence, dtype=np.float32)
        if confidence.shape != (size,) or not np.isfinite(confidence).all():
            raise ValueError(f"confidence must be finite with shape ({size},)")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence values must be between 0 and 1")
        finite_anchors = np.isfinite(wrist).all(axis=1) & np.isfinite(middle).all(axis=1)
        if np.any(valid & ~finite_anchors):
            raise ValueError("valid observations must contain finite camera-space anchors")

        area = self.projected_area
        if area is not None:
            area = np.asarray(area, dtype=np.float32)
            if area.shape != (size,) or np.any(~np.isfinite(area)) or np.any(area < 0):
                raise ValueError(f"projected_area must be finite, non-negative, and have shape ({size},)")

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if self.coordinate_frame != CAMERA_FRAME:
            raise ValueError("coordinate_frame must be 'camera'")
        if self.units != METERS:
            raise ValueError("units must be 'meters'")

        keys = np.stack((clip_id, chunk_id, hand_side), axis=1)
        for key in np.unique(keys, axis=0):
            mask = np.all(keys == key, axis=1)
            frames = frame_index[mask]
            if len(np.unique(frames)) != len(frames):
                joined = "/".join(key.tolist())
                raise ValueError(f"duplicate frame_index within chunk {joined}")

        object.__setattr__(self, "clip_id", clip_id)
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "frame_index", frame_index.astype(np.int64, copy=False))
        object.__setattr__(self, "hand_side", hand_side)
        object.__setattr__(self, "wrist_cam", wrist)
        object.__setattr__(self, "middle_mcp_cam", middle)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "projected_area", area)

    def __len__(self) -> int:
        return len(self.frame_index)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "HandObservations":
        required = (
            "schema_version", "coordinate_frame", "units", "clip_id", "chunk_id",
            "frame_index", "hand_side", "wrist_cam", "middle_mcp_cam", "valid",
            "confidence",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        return cls(
            schema_version=_scalar_string(values["schema_version"], "schema_version"),
            coordinate_frame=_scalar_string(values["coordinate_frame"], "coordinate_frame"),
            units=_scalar_string(values["units"], "units"),
            clip_id=values["clip_id"],
            chunk_id=values["chunk_id"],
            frame_index=values["frame_index"],
            hand_side=values["hand_side"],
            wrist_cam=values["wrist_cam"],
            middle_mcp_cam=values["middle_mcp_cam"],
            valid=values["valid"],
            confidence=values["confidence"],
            projected_area=values.get("projected_area"),
        )

    def to_mapping(self) -> dict[str, np.ndarray]:
        values = {
            "schema_version": np.asarray(self.schema_version),
            "coordinate_frame": np.asarray(self.coordinate_frame),
            "units": np.asarray(self.units),
            "clip_id": self.clip_id,
            "chunk_id": self.chunk_id,
            "frame_index": self.frame_index,
            "hand_side": self.hand_side,
            "wrist_cam": self.wrist_cam,
            "middle_mcp_cam": self.middle_mcp_cam,
            "valid": self.valid,
            "confidence": self.confidence,
        }
        if self.projected_area is not None:
            values["projected_area"] = self.projected_area
        return values


def load_observations_npz(path: str | Path) -> HandObservations:
    with np.load(path, allow_pickle=False) as data:
        return HandObservations.from_mapping(data)


def save_observations_npz(path: str | Path, observations: HandObservations) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **observations.to_mapping())

