#!/usr/bin/env python3
"""Run one HandsOnWorld sample with checkpoints matching the configured backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from handsonworld.inference_batch import checkpoint_paths  # noqa: E402
from handsonworld.training.config import (  # noqa: E402
    ConfigError, load_config, model_levels, model_spec, resolve_path,
)
from handsonworld.training.metadata import validate_weight_metadata  # noqa: E402
from handsonworld.training.runtime import add_backend_to_path, run_inference, validate_resources  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--high-checkpoint")
    result.add_argument("--low-checkpoint")
    result.add_argument("--checkpoint", help="Checkpoint override for single-DiT models")
    result.add_argument("--epoch", type=int, help="Zero-based epoch; defaults to the final training epoch")
    result.add_argument("--dry-run", action="store_true", help="Validate config without loading model weights")
    result.add_argument("--allow-missing-metadata", action="store_true", help="For legacy private weights only")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        config = load_config(args.config, "infer")
        levels = model_levels(config)
        checkpoints = checkpoint_paths(
            config, args.config, args.high_checkpoint, args.low_checkpoint, args.epoch, args.checkpoint
        )
        for level, path in checkpoints.items():
            key = "checkpoint" if level == "single" else f"{level}_checkpoint"
            config["inference"][key] = path
        summary = {
            "model_type": model_spec(config)["model_type"],
            "representation": config["representation"],
            "control_in_dim": 48,
            "channel_order": config["control"]["channel_order"],
            "timestep_ranges": {
                level: [
                    config["models"][level]["min_timestep_boundary"],
                    config["models"][level]["max_timestep_boundary"],
                ]
                for level in levels
            },
            "checkpoints": checkpoints,
            "switch_dit_boundary": config["inference"].get("switch_dit_boundary"),
        }
        print(json.dumps(summary, indent=2))
        if args.dry_run:
            return 0
        add_backend_to_path(config, args.config)
        validate_resources(config, args.config, "infer")
        for level in levels:
            validate_weight_metadata(
                resolve_path(args.config, config["models"][level]["adapter"]),
                config["representation"], level,
                model_type=model_spec(config)["model_type"],
                required=not args.allow_missing_metadata, allow_initializer=True,
            )
            checkpoint_key = "checkpoint" if level == "single" else f"{level}_checkpoint"
            validate_weight_metadata(
                resolve_path(args.config, config["inference"][checkpoint_key]),
                config["representation"], level,
                model_type=model_spec(config)["model_type"],
                required=not args.allow_missing_metadata,
            )
        output = run_inference(config, args.config)
        print(f"Wrote {output}")
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
