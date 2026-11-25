#!/usr/bin/env python3
"""Prepare a ``video,prompt`` CSV for HandsOnWorld training with a hand estimator and protagonist filtering."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from handsonworld.data.videos import (  # noqa: E402
    build_preprocessing_command,
    default_python,
    default_runner,
    execute_and_collect,
    load_video_inputs,
    write_normalized_input,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path, help="CSV with video,prompt columns")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hawor-root", type=Path, help="external HaWoR checkout")
    parser.add_argument("--estimator", default="hawor", help="hawor or importable module:Class")
    parser.add_argument("--estimator-config", type=Path, help="JSON constructor options for the estimator")
    parser.add_argument(
        "--backend-manifest",
        type=Path,
        help="collect an existing runner manifest instead of rerunning estimation and filtering",
    )
    parser.add_argument("--runner", type=Path, default=default_runner(REPO_ROOT))
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--sampling", choices=("chunk_first", "all_frames"), default="chunk_first")
    parser.add_argument("--min-pass-ratio", type=float, default=0.9)
    parser.add_argument(
        "--smpl-model-path", type=Path, default=REPO_ROOT / "models" / "body_models"
    )
    parser.add_argument(
        "--vposer-path", type=Path, default=REPO_ROOT / "models" / "vposer_v1_0"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-iterations", type=int, default=500)
    parser.add_argument("--fit-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--fill-mode", choices=("interpolation", "network"), default="interpolation"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = load_video_inputs(args.input_csv)
    output_dir = args.output_dir.expanduser().resolve()
    normalized_csv = output_dir / "input.normalized.csv"
    backend_dir = output_dir / ("hawor" if args.estimator == "hawor" else "estimator")
    backend_manifest = args.backend_manifest

    command: list[str] | None = None
    if backend_manifest is None:
        if args.estimator == "hawor" and args.hawor_root is None and args.estimator_config is None:
            parser.error("--hawor-root is required for the HaWoR estimator")
        if args.estimator_config is not None and not args.estimator_config.is_file():
            parser.error(f"Estimator config does not exist: {args.estimator_config}")
        command = build_preprocessing_command(
            python=args.python,
            runner=args.runner.expanduser().resolve(),
            input_csv=normalized_csv,
            output_dir=backend_dir,
            hawor_root=args.hawor_root.expanduser().resolve() if args.hawor_root else None,
            estimator=args.estimator,
            estimator_config=args.estimator_config.expanduser().resolve() if args.estimator_config else None,
            sampling=args.sampling,
            min_pass_ratio=args.min_pass_ratio,
            smpl_model_path=args.smpl_model_path.expanduser().resolve(),
            vposer_path=args.vposer_path.expanduser().resolve(),
            device=args.device,
            fit_iterations=args.fit_iterations,
            fit_learning_rate=args.fit_learning_rate,
            fill_mode=args.fill_mode,
        )
        backend_manifest = backend_dir / "manifest.csv"

    print(f"Inputs: {len(inputs)}")
    print(f"Sampling: {args.sampling}; strict pass ratio > {args.min_pass_ratio}; beta: chunk_shared")
    if args.dry_run:
        if command is None:
            print(f"Collect backend manifest: {backend_manifest.expanduser().resolve()}")
        else:
            print(f"Estimator command ({args.estimator}):")
            print(shlex.join(command))
        print(f"Training CSV: {output_dir / 'train.csv'}")
        print(f"Processing report: {output_dir / 'processing_report.csv'}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_normalized_input(normalized_csv, inputs)
    if command is not None:
        if not args.runner.is_file():
            parser.error(f"Preprocessing runner does not exist: {args.runner}")
    _, summary = execute_and_collect(
        inputs,
        backend_manifest.expanduser().resolve(),
        command=command,
        train_csv=output_dir / "train.csv",
        report_csv=output_dir / "processing_report.csv",
    )
    print(
        f"Prepared {summary.successes} training rows; {summary.rejected} filtered; "
        f"{summary.errors} errors (see processing_report.csv)"
    )
    return 2 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
