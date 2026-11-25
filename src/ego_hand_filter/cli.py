"""Command-line interface for offline hand candidate filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .adapters.hawor import load_camera_joints_npz
from .decision import decision_mapping, filter_chunks
from .schema import load_observations_npz, save_observations_npz


def _ratio(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("ratio must be finite and between 0 and 1")
    return result


def _synthetic_checker(anchors: np.ndarray, hand_side: str):
    """Deterministic test/demo checker; this is not the paper body fitter."""
    passed = anchors[:, 0, 0] > 0
    return passed, passed.astype(np.float32), {"checker": "synthetic_x_positive"}


def _write_decisions(path: Path, decisions, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload = {
            "report_version": "ego-hand-filter-report-v1",
            "settings": settings,
            "chunks": [decision.to_dict() for decision in decisions],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif path.suffix.lower() == ".npz":
        np.savez_compressed(path, settings_json=json.dumps(settings), **decision_mapping(decisions))
    else:
        raise ValueError("output path must end in .json or .npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ego-hand-filter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser(
        "convert-hawor-npz", help="convert a pre-filter HaWoR camera-joints export"
    )
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)

    run = subparsers.add_parser("filter", help="filter standard-schema NPZ observations")
    run.add_argument("input", type=Path)
    run.add_argument("output", type=Path, help="decision report (.json or .npz)")
    run.add_argument("--sampling", choices=("chunk_first", "all_frames"), default="chunk_first")
    run.add_argument("--min-pass-ratio", type=_ratio, default=0.9)
    run.add_argument(
        "--checker",
        choices=("body", "synthetic_x_positive"),
        default="body",
        help="body is the paper method; synthetic_x_positive is only for schema/demo checks",
    )
    run.add_argument("--beta-mode", choices=("chunk_shared", "per_frame"), default="chunk_shared")
    run.add_argument("--smpl-model-path", type=Path)
    run.add_argument("--vposer-path", type=Path)
    run.add_argument("--device", default="cpu")
    run.add_argument("--fit-iterations", type=int, default=500)
    run.add_argument("--fit-learning-rate", type=float, default=0.05)
    run.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert-hawor-npz":
        save_observations_npz(args.output, load_camera_joints_npz(args.input))
        return 0

    observations = load_observations_npz(args.input)
    if args.checker == "synthetic_x_positive":
        checker = _synthetic_checker
    else:
        if args.smpl_model_path is None or args.vposer_path is None:
            raise SystemExit("body checker requires --smpl-model-path and --vposer-path")
        if args.fit_iterations < 0:
            raise SystemExit("--fit-iterations must be non-negative")
        from .fitting import SMPLBodyFitter

        fitter = SMPLBodyFitter(
            smpl_model_path=args.smpl_model_path,
            vposer_path=args.vposer_path,
            device=args.device,
        )
        checker = fitter.make_checker(
            n_iter=args.fit_iterations,
            lr=args.fit_learning_rate,
            beta_mode=args.beta_mode,
            verbose=args.verbose,
        )
    decisions = filter_chunks(
        observations,
        checker,
        sampling=args.sampling,
        min_pass_ratio=args.min_pass_ratio,
    )
    _write_decisions(args.output, decisions, {
        "checker": args.checker, "sampling": args.sampling,
        "min_pass_ratio": args.min_pass_ratio, "comparison": ">",
        "beta_mode": args.beta_mode, "loss_reduction": "sum",
        "fit_iterations": args.fit_iterations, "fit_learning_rate": args.fit_learning_rate,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

