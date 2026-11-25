"""Chunk sampling and strict pass-ratio aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np

from .schema import HandObservations


Checker = Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray, dict]]


@dataclass(frozen=True)
class ChunkDecision:
    clip_id: str
    chunk_id: str
    hand_side: str
    frame_indices: list[int]
    sampled_frame_indices: list[int]
    passed: list[bool]
    confidence: list[float]
    pass_ratio: float
    keep: bool
    reason: str
    sampling: str
    min_pass_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_options(sampling: str, min_pass_ratio: float) -> None:
    if sampling not in {"chunk_first", "all_frames"}:
        raise ValueError("sampling must be 'chunk_first' or 'all_frames'")
    if not np.isfinite(min_pass_ratio) or not 0 <= min_pass_ratio <= 1:
        raise ValueError("min_pass_ratio must be finite and between 0 and 1")


def decide_chunk(
    anchors: np.ndarray,
    frame_indices: np.ndarray,
    valid: np.ndarray,
    hand_side: str,
    checker: Checker,
    *,
    sampling: str = "chunk_first",
    min_pass_ratio: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, float, bool, np.ndarray]:
    """Return per-sample results and a strict chunk-level decision."""
    _validate_options(sampling, min_pass_ratio)
    anchors = np.asarray(anchors, dtype=np.float32)
    frames = np.asarray(frame_indices, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    if anchors.ndim != 3 or anchors.shape[1:] != (2, 3):
        raise ValueError("anchors must have shape (T, 2, 3)")
    if frames.shape != (len(anchors),) or valid.shape != (len(anchors),):
        raise ValueError("frame_indices and valid must have one value per anchor")
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")

    sample_positions = np.arange(len(anchors), dtype=np.int64)
    if sampling == "chunk_first":
        sample_positions = sample_positions[:1]
    sample_anchors = anchors[sample_positions]
    usable = valid[sample_positions] & np.isfinite(sample_anchors).all(axis=(1, 2))
    passed = np.zeros(len(sample_positions), dtype=bool)
    confidence = np.zeros(len(sample_positions), dtype=np.float32)
    if usable.any():
        checked_passed, checked_confidence, _ = checker(sample_anchors[usable], hand_side)
        checked_passed = np.asarray(checked_passed, dtype=bool)
        checked_confidence = np.asarray(checked_confidence, dtype=np.float32)
        expected = (int(usable.sum()),)
        if checked_passed.shape != expected or checked_confidence.shape != expected:
            raise ValueError("checker must return one decision and confidence per input frame")
        if not np.isfinite(checked_confidence).all():
            raise ValueError("checker confidence must be finite")
        passed[usable] = checked_passed
        confidence[usable] = checked_confidence
    pass_ratio = float(passed.mean()) if len(passed) else 0.0
    keep = bool(len(passed) and pass_ratio > min_pass_ratio)
    return passed, confidence, pass_ratio, keep, sample_positions


def filter_chunks(
    observations: HandObservations,
    checker: Checker,
    *,
    sampling: str = "chunk_first",
    min_pass_ratio: float = 0.9,
) -> list[ChunkDecision]:
    """Filter each clip/chunk/hand group independently."""
    _validate_options(sampling, min_pass_ratio)
    group_keys = list(zip(observations.clip_id, observations.chunk_id, observations.hand_side))
    ordered_keys = list(dict.fromkeys(group_keys))
    decisions: list[ChunkDecision] = []
    for clip_id, chunk_id, hand_side in ordered_keys:
        mask = (
            (observations.clip_id == clip_id)
            & (observations.chunk_id == chunk_id)
            & (observations.hand_side == hand_side)
        )
        rows = np.flatnonzero(mask)
        rows = rows[np.argsort(observations.frame_index[rows], kind="stable")]
        anchors = np.stack(
            (observations.wrist_cam[rows], observations.middle_mcp_cam[rows]), axis=1
        )
        passed, confidence, ratio, keep, sampled = decide_chunk(
            anchors,
            observations.frame_index[rows],
            observations.valid[rows],
            str(hand_side),
            checker,
            sampling=sampling,
            min_pass_ratio=min_pass_ratio,
        )
        decisions.append(
            ChunkDecision(
                clip_id=str(clip_id),
                chunk_id=str(chunk_id),
                hand_side=str(hand_side),
                frame_indices=observations.frame_index[rows].tolist(),
                sampled_frame_indices=observations.frame_index[rows][sampled].tolist(),
                passed=passed.tolist(),
                confidence=confidence.tolist(),
                pass_ratio=ratio,
                keep=keep,
                reason="passed" if keep else "pass_ratio_not_strictly_above_threshold",
                sampling=sampling,
                min_pass_ratio=float(min_pass_ratio),
            )
        )
    return decisions


def decision_mapping(decisions: Iterable[ChunkDecision]) -> dict[str, np.ndarray]:
    """Create a compact NPZ-compatible chunk report."""
    items = list(decisions)
    return {
        "clip_id": np.asarray([item.clip_id for item in items]),
        "chunk_id": np.asarray([item.chunk_id for item in items]),
        "hand_side": np.asarray([item.hand_side for item in items]),
        "pass_ratio": np.asarray([item.pass_ratio for item in items], dtype=np.float32),
        "keep": np.asarray([item.keep for item in items], dtype=bool),
        "sampling": np.asarray([item.sampling for item in items]),
        "min_pass_ratio": np.asarray([item.min_pass_ratio for item in items], dtype=np.float32),
    }

