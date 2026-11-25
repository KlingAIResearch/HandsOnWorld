"""Temporal boundaries shared by causal attention, training and generation."""

import torch


def temporal_chunk_ranges(num_frames: int, chunk_size: int, fixed_first_frame: bool = False):
    """Return exclusive frame ranges, preserving legacy front-loaded remainder.

    A fixed first frame is its own clean prefix; all following chunks must be
    complete. Frame counts are latent frames, not decoded video frames.
    """
    if num_frames < 1 or chunk_size < 1:
        raise ValueError("num_frames and chunk_size must be positive")
    if fixed_first_frame:
        if (num_frames - 1) % chunk_size:
            raise ValueError("With a fixed first frame, (num_frames - 1) must be divisible by chunk_size")
        return [(0, 1)] + [(start, start + chunk_size) for start in range(1, num_frames, chunk_size)]
    if num_frames < chunk_size:
        return [(0, num_frames)]
    first_end = chunk_size + num_frames % chunk_size
    return [(0, first_end)] + [(start, start + chunk_size) for start in range(first_end, num_frames, chunk_size)]


def clean_first_frame(condition, reference):
    """Read the same I2V prefix in training and inference, without gradients."""
    if not isinstance(condition, dict) or "first_frame_cond" not in condition:
        raise ValueError("Fixed-first-frame generation requires first_frame_cond")
    value = condition["first_frame_cond"]
    if not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[2] < 1:
        raise ValueError("first_frame_cond must have shape [B,C,T,H,W] with T >= 1")
    prefix = value[:, :, :1]
    if prefix.shape != reference[:, :, :1].shape:
        raise ValueError("first_frame_cond batch/channel/spatial shape differs from video latents")
    return prefix.detach().to(device=reference.device, dtype=reference.dtype)
