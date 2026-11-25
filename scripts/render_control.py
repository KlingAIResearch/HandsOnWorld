#!/usr/bin/env python3
"""Render the same 12-channel hand/camera map used by the training pipeline."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "DiffSynth-Studio"))


def load_inputs(hand_path, camera_path, intrinsics_path, faces_path):
    hand = np.load(hand_path, allow_pickle=False).astype(np.float32)
    camera = np.load(camera_path, allow_pickle=False).astype(np.float32)
    intrinsics = np.load(intrinsics_path, allow_pickle=False).astype(np.float32)
    faces = np.loadtxt(faces_path, dtype=np.int64, ndmin=2)
    if hand.ndim != 3 or hand.shape[-1] != 3 or len(hand) == 0:
        raise ValueError("hand vertices must have shape [T, V, 3] with T > 0")
    if camera.shape != (len(hand), 4, 4):
        raise ValueError("camera must contain one world-to-camera [4,4] transform per hand frame")
    if intrinsics.shape != (3, 3) or np.linalg.det(intrinsics) == 0:
        raise ValueError("intrinsics must be one invertible [3,3] matrix")
    if faces.shape[1] != 3 or faces.size == 0 or faces.min() < 0 or faces.max() >= hand.shape[1]:
        raise ValueError("faces must be valid zero-based vertex-index triangles")
    if not all(np.isfinite(value).all() for value in (hand, camera, intrinsics)):
        raise ValueError("control inputs must be finite")
    if not np.allclose(camera[:, 3], [0, 0, 0, 1]):
        raise ValueError("camera matrices must have homogeneous last row [0,0,0,1]")
    return hand, camera, intrinsics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-vertices", required=True)
    parser.add_argument("--camera", required=True, help="world-to-camera transforms in the same metric world frame")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--faces", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--original-height", type=int, required=True)
    parser.add_argument("--original-width", type=int, required=True)
    parser.add_argument("--mode", choices=("normal_plucker", "reflection_plucker"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.height, args.width, args.original_height, args.original_width) <= 0:
        parser.error("all image dimensions must be positive")
    hand, camera, intrinsics = load_inputs(args.hand_vertices, args.camera, args.intrinsics, args.faces)
    contract = {"mode": args.mode, "channel_order": "hand_then_camera", "space": "clip_start_world",
                "shape": [1, 12, len(hand), args.height, args.width]}
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return

    import torch
    from types import SimpleNamespace
    from diffsynth.pipelines.wan_video import (
        WanVideoUnit_RebasePoseToClipStart, WanVideoUnit_HandPosePluckerControlAdapter,
    )
    from diffsynth.pipelines.plucker_utils import poses_to_plucker_rays
    from handsonworld.conditioning.control import combine_control_maps

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering requires CUDA and nvdiffrast; use --dry-run to validate inputs on CPU")
    device = torch.device("cuda")
    pipe = SimpleNamespace(device=device, torch_dtype=torch.float32)
    with torch.no_grad():
        rebased = WanVideoUnit_RebasePoseToClipStart().process(pipe, None, camera, hand)
        K = torch.as_tensor(intrinsics, device=device)
        camera_map = poses_to_plucker_rays(
            rebased["cam_pose"][0], orig_height=args.original_height, orig_width=args.original_width,
            target_height=args.height, target_width=args.width, camera_intrinsics=K,
        ).unsqueeze(0)
        unit = WanVideoUnit_HandPosePluckerControlAdapter(faces_path=args.faces)
        hand_map = unit._render_pose_latents(
            rebased["hand_pose"], rebased["cam_pose"], args.mode, "world", args.height, args.width,
            args.original_height, args.original_width, 32, device, camera_intrinsics=K,
        )
        control = combine_control_maps(hand_map, camera_map).cpu().numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, control=control, metadata=json.dumps(contract))
    print(f"Saved {control.shape} to {output}")


if __name__ == "__main__":
    main()
