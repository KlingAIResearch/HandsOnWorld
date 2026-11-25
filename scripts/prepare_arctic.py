#!/usr/bin/env python3
"""Prepare official ARCTIC ego-view data as a HandsOnWorld training CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from handsonworld.data.arctic import (  # noqa: E402
    export_hand_vertices,
    export_video,
    iter_plans,
    load_mano_layers,
    processed_intrinsics,
    relative_world_to_camera,
    resolve_data_root,
    resolve_mano_dir,
    sequence_prompt,
    sequence_settings,
    validate_existing_outputs,
)
from handsonworld.data.manifest import write_manifest  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-dir", required=True, type=Path, help="Official ARCTIC unpack root or arctic_data/data")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--mano-dir", type=Path, help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl")
    result.add_argument("--protocol", choices=("p1", "p2"), default="p2")
    result.add_argument("--split", choices=("train", "val", "test"), default="train")
    result.add_argument("--frame-stride", type=int, default=2)
    result.add_argument("--skip-initial-frames", type=int, default=2)
    result.add_argument("--min-num-frames", type=int, default=81)
    result.add_argument("--fps", type=float, default=30.0)
    result.add_argument("--device", default="cuda")
    result.add_argument("--max-sequences", type=int)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true", help="Inspect and validate inputs without writing files or loading MANO")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.frame_stride < 1 or args.skip_initial_frames < 0 or args.min_num_frames < 1:
        raise SystemExit("frame-stride and min-num-frames must be positive; skip-initial-frames must be non-negative")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise SystemExit("fps must be finite and positive")
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise SystemExit("max-sequences must be positive")
    try:
        data_root = resolve_data_root(args.base_dir)
        mano_dir = resolve_mano_dir(args.base_dir, data_root, args.mano_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    report = []
    plans = iter_plans(
        data_root, args.protocol, args.split, args.skip_initial_frames,
        args.frame_stride, issues=report,
    )
    selected = []
    for plan in plans:
        if len(plan[1]) >= args.min_num_frames:
            selected.append(plan)
        else:
            report.append({
                "subject": plan[0].subject, "sequence": plan[0].sequence,
                "split": plan[0].split, "status": "short",
                "message": f"num_frames={len(plan[1])} < {args.min_num_frames}",
            })
        if args.max_sequences is not None and len(selected) >= args.max_sequences:
            break

    print(f"ARCTIC data: {data_root}")
    print(f"MANO models: {mano_dir}")
    print(f"Selected {len(selected)} {args.protocol}/{args.split} ego-view sequences")
    if selected:
        image_width, image_height = selected[0][6]
        left, top, right, bottom = selected[0][7]
        print(
            f"Ego images: {image_width}x{image_height}; "
            f"center-cropped output: {right - left}x{bottom - top} (4:3)"
        )
    if args.dry_run:
        return 0

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("ARCTIC export requires PyTorch; install the fitting extras") from exc
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    mano_layers = None
    output_dir = args.output_dir.expanduser().resolve()
    rows = []
    for sequence, paths, indices, world_to_camera, intrinsics, source_size, image_size, crop_box in selected:
        control_dir = output_dir / "controls" / sequence.subject / sequence.sequence
        video_path = output_dir / "videos" / sequence.subject / f"{sequence.sequence}.mp4"
        hand_path = control_dir / "hand_vertices.npy"
        camera_path = control_dir / "camera.npy"
        intrinsics_path = control_dir / "intrinsics.npy"
        settings_path = control_dir / "settings.json"
        selected_world_to_camera = world_to_camera[indices]
        settings = sequence_settings(
            sequence, indices, image_size, crop_box, args.protocol,
            args.skip_initial_frames, args.frame_stride, args.fps,
        )
        existing = any(path.exists() for path in (
            video_path, hand_path, camera_path, intrinsics_path, settings_path,
        ))
        if existing and not args.overwrite:
            try:
                validate_existing_outputs(
                    video_path, hand_path, camera_path, intrinsics_path,
                    settings_path, settings,
                )
            except (OSError, ValueError) as exc:
                raise SystemExit(str(exc)) from exc
            action = "reused"
        else:
            if mano_layers is None:
                mano_layers = load_mano_layers(mano_dir, device)
            export_video(paths, video_path, crop_box, args.fps / args.frame_stride)
            hand_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(hand_path, export_hand_vertices(sequence.mano_path, indices, selected_world_to_camera, mano_layers, device))
            camera_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(camera_path, relative_world_to_camera(selected_world_to_camera))
            np.save(intrinsics_path, processed_intrinsics(intrinsics, source_size, image_size, crop_box))
            settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
            action = "generated"
        rows.append({
            "video": str(video_path), "prompt": sequence_prompt(sequence.sequence), "poses": "", "pose_video": "",
            "cam_pose": str(camera_path), "hand_pose": str(hand_path), "camera_intrinsics": str(intrinsics_path),
            "width": crop_box[2] - crop_box[0], "height": crop_box[3] - crop_box[1], "num_frames": len(paths),
            "allow_random_start_frame": 1, "source": "arctic", "status": "success", "fps": args.fps / args.frame_stride,
            "frame_stride": args.frame_stride, "skip_initial_frames": args.skip_initial_frames,
            "subject": sequence.subject, "sequence": sequence.sequence, "split": sequence.split, "message": "",
        })
        report.append({
            "subject": sequence.subject, "sequence": sequence.sequence,
            "split": sequence.split, "status": "success", "message": action,
        })
    manifest = output_dir / "train.csv"
    write_manifest(manifest, rows)
    report_path = output_dir / "processing_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject", "sequence", "split", "status", "message"))
        writer.writeheader()
        writer.writerows(report)
    print(f"Wrote {manifest} ({len(rows)} rows)")
    print(f"Wrote {report_path} ({len(report)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
