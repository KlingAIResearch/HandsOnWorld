"""Keep the training checkpoint's hand-first channel order explicit."""

import torch

CHANNEL_ORDER = "hand_then_camera"


def combine_control_maps(hand_map, camera_map):
    """Concatenate [B, C, T, H, W] control maps as hand then camera.

    Temporal packing from 12 to 48 channels remains in the pinned backend,
    including its first-frame repetition and visibility-mask handling.
    """
    return torch.cat([hand_map, camera_map], dim=1)
