#!/usr/bin/env python3
"""Prepare Wan DiTs with control adapters, independently of datasets and encoders."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handsonworld.training.config import (  # noqa: E402
    CONTROL_CHANNEL_ORDER, MODEL_ALIASES, NOISE_RANGES, load_config,
    model_spec, resolve_noise_levels, resolve_path, validate_config,
)
from handsonworld.training.metadata import validate_weight_metadata  # noqa: E402
from handsonworld.training.resources import prepare_resources  # noqa: E402


def model_config(model, model_dir=None, output_dir=None):
    """Build an initializer-only contract without reading a training recipe."""
    spec = model_spec({"model_type": MODEL_ALIASES[model]})
    base = (model_dir or ROOT / "models" / spec["model_id"].split("/")[-1]).expanduser().resolve()
    output = (output_dir or ROOT / "models/initialized" / model).expanduser().resolve()
    models = {}
    for level in spec["levels"]:
        shard_dir = base / f"{level}_noise_model" if level != "single" else base
        low, high = NOISE_RANGES[level] if level != "single" else (0.0, 1.0)
        models[level] = {
            "source_shards": [str(shard_dir / f"diffusion_pytorch_model-{i:05d}-of-{spec['num_shards']:05d}.safetensors")
                              for i in range(1, spec["num_shards"] + 1)],
            "adapter": str(output / f"{level}.safetensors"),
            "min_timestep_boundary": low, "max_timestep_boundary": high,
        }
    return {"model_type": spec["model_type"], "backend_root": str(ROOT / "third_party/DiffSynth-Studio"),
            "control": {"in_dim": 48, "attr_space": "world", "channel_order": list(CONTROL_CHANNEL_ORDER)},
            "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", choices=tuple(MODEL_ALIASES))
    source.add_argument("--config", help="Legacy/custom JSON paths; representation does not affect initialization")
    parser.add_argument("--model-dir", type=Path, help="Base Wan model directory (with --model)")
    parser.add_argument("--output-dir", type=Path, help="Initialized DiT directory (with --model)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for newly initialized adapters")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading or loading weights")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--reuse-root", type=Path, action="append")
    parser.add_argument("--noise-level", choices=("high", "low", "both", "single", "auto"), default="both")
    args = parser.parse_args()
    try:
        if args.config and (args.model_dir or args.output_dir):
            parser.error("--model-dir and --output-dir require --model; use JSON paths with --config")
        config_path = args.config or ROOT / "configs/initializer.json"
        config = (load_config(args.config, "initialize") if args.config else
                  model_config(args.model, args.model_dir, args.output_dir))
        validate_config(config, "initialize")
        levels = resolve_noise_levels(config, args.noise_level)
        if args.dry_run:
            print(json.dumps({"model_type": model_spec(config)["model_type"], "seed": args.seed,
                              "models": {level: config["models"][level] for level in levels}}, indent=2))
            return
        report = prepare_resources(config, config_path, levels=levels, offline=args.offline,
                                   reuse_roots=args.reuse_root, asset_keys=(), seed=args.seed)
        for level in levels:
            validate_weight_metadata(
                resolve_path(config_path, config["models"][level]["adapter"]),
                config.get("representation", "normal_plucker"), level,
                model_type=model_spec(config)["model_type"], allow_initializer=True,
            )
        print(json.dumps(report, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
