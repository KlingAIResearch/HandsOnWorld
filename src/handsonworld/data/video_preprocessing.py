"""Shared estimator loading, body fitting and batch orchestration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import traceback
from typing import Callable

import numpy as np

from .estimators import FilteredOutError, HandEstimator, VideoProcessor
from .videos import BACKEND_FIELDS as MANIFEST_FIELDS


def load_estimator(args: argparse.Namespace) -> HandEstimator:
    options = {}
    if args.estimator_config is not None:
        options = json.loads(args.estimator_config.read_text(encoding="utf-8"))
        if not isinstance(options, dict):
            raise ValueError("--estimator-config must contain a JSON object")
    if "device" in options:
        raise ValueError("Set the estimator device with --device, not estimator-config")
    if "min_num_frames" in options:
        raise ValueError("Set --min-num-frames on the runner; estimator-config cannot override the shared policy")
    if args.estimator == "hawor":
        from .hawor_runner import HaWoREstimator

        defaults = {
            "checkpoint": args.checkpoint, "detector": args.detector,
            "infiller_weight": args.infiller_weight, "img_focal": args.img_focal,
            "fill_mode": args.fill_mode, "slam_filter_threshold": args.slam_filter_threshold,
            "min_num_frames": args.min_num_frames,
        }
        defaults.update(options)
        if args.hawor_root is not None:
            defaults["hawor_root"] = args.hawor_root
        if not defaults.get("hawor_root"):
            raise ValueError("--hawor-root is required for the HaWoR estimator")
        return HaWoREstimator(device=args.device, **defaults)
    module, separator, name = args.estimator.partition(":")
    if not separator or not module or not name:
        raise ValueError("--estimator must be hawor or an importable module:Class")
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, HandEstimator):
        raise TypeError("Estimator class must inherit handsonworld.data.estimators.HandEstimator")
    return cls(device=args.device, **options)


def make_checker(args: argparse.Namespace):
    from ego_hand_filter.fitting import SMPLBodyFitter

    fitter = SMPLBodyFitter(
        smpl_model_path=args.smpl_model_path, vposer_path=args.vposer_path,
        device=args.device,
    )
    return fitter.make_checker(
        n_iter=args.fit_iterations, lr=args.fit_learning_rate, beta_mode=args.beta_mode,
    )


def run_manifest(
    input_csv: Path, output_dir: Path, processor: Callable[[Path, str, Path], dict]
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"video", "prompt"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"input CSV is missing columns: {sorted(missing)}")
        inputs = list(reader)
    rows = []
    for item in inputs:
        video = Path(item["video"]).expanduser()
        if not video.is_absolute():
            video = (input_csv.parent / video).resolve()
        prompt = item["prompt"]
        try:
            if not video.is_file():
                raise FileNotFoundError(video)
            row = processor(video, prompt, output_dir)
        except Exception as error:  # keep batch processing and make failures inspectable
            status = "filtered" if isinstance(error, FilteredOutError) else "failed"
            message = f"{type(error).__name__}: {error}"
            if status == "failed":
                error_dir = output_dir / "errors"
                error_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha1(str(video).encode()).hexdigest()[:10]
                error_path = error_dir / f"{video.stem}-{digest}.log"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                message += f" (traceback: {error_path.resolve()})"
            row = {
                "input_video": str(video), "video": "", "prompt": prompt,
                "cam_pose": "", "hand_pose": "",
                "camera_intrinsics": "", "width": "", "height": "", "num_frames": "",
                "status": status, "message": message,
            }
        rows.append(row)
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in MANIFEST_FIELDS} for row in rows])
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a hand estimator and the shared protagonist-hand filter")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path)
    parser.add_argument("--estimator", default="hawor", help="hawor or importable module:Class")
    parser.add_argument("--estimator-config", type=Path, help="JSON constructor options for the estimator")
    parser.add_argument("--sampling", choices=("chunk_first", "all_frames"), default="chunk_first")
    parser.add_argument("--min-pass-ratio", type=float, default=0.9)
    parser.add_argument("--beta-mode", choices=("chunk_shared", "per_frame"), default="chunk_shared")
    parser.add_argument("--smpl-model-path", type=Path, required=True)
    parser.add_argument("--vposer-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--infiller-weight", type=Path)
    parser.add_argument("--img-focal", type=float)
    parser.add_argument("--min-num-frames", type=int, default=81)
    parser.add_argument("--fit-iterations", type=int, default=500)
    parser.add_argument("--fit-learning-rate", type=float, default=0.05)
    parser.add_argument("--fill-mode", choices=("interpolation", "network"), default="interpolation")
    parser.add_argument("--slam-filter-threshold", type=float, default=1.0)
    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not np.isfinite(args.min_pass_ratio) or not 0 <= args.min_pass_ratio <= 1:
        parser.error("--min-pass-ratio must be finite and between 0 and 1")
    if args.min_num_frames <= 0:
        parser.error("--min-num-frames must be positive")
    if args.fit_iterations < 0 or not np.isfinite(args.fit_learning_rate) or args.fit_learning_rate <= 0:
        parser.error("fit iterations must be nonnegative and learning rate finite and positive")
    # Check constructor/contract before allocating the fitting models.
    estimator = load_estimator(args)
    processor = VideoProcessor(
        estimator, make_checker(args), sampling=args.sampling,
        min_pass_ratio=args.min_pass_ratio, min_num_frames=args.min_num_frames,
        provenance={
            "estimator": args.estimator, "beta_mode": args.beta_mode,
            "device": args.device, "fit_iterations": args.fit_iterations,
            "fit_learning_rate": args.fit_learning_rate,
            "estimator_options": getattr(estimator, "provenance", {}),
            "estimator_config": (
                json.loads(args.estimator_config.read_text(encoding="utf-8"))
                if args.estimator_config else {}
            ),
        },
    )
    rows = run_manifest(args.input_csv.resolve(), args.output_dir.resolve(), processor)
    success = sum(row["status"] == "success" for row in rows)
    filtered = sum(row["status"] == "filtered" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"Video preprocessing: {success} succeeded, {filtered} filtered, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
