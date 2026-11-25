#!/usr/bin/env python3
"""Add a representation-independent 48-channel control adapter to configured Wan DiTs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from handsonworld.training.config import (  # noqa: E402
    ConfigError, load_config, model_levels, model_spec, resolve_path,
)
from handsonworld.training.metadata import write_initializer_metadata  # noqa: E402
from handsonworld.training.runtime import add_backend_to_path, initialize_adapter  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true", help="Validate and print outputs without model weights")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        config = load_config(args.config, "initialize")
        spec = model_spec(config)
        plans = []
        for level in model_levels(config):
            model = config["models"][level]
            plans.append({
                "noise_level": level,
                "source_shards": [str(resolve_path(args.config, item)) for item in model["source_shards"]],
                "output": str(resolve_path(args.config, model["adapter"])),
                "seed": args.seed,
                "in_dim": 48,
                "model_dim": spec["dim"],
                "downscale_factor": spec["downscale_factor"],
            })
        print(json.dumps(plans, indent=2))
        if args.dry_run:
            return 0
        add_backend_to_path(config, args.config)
        for plan in plans:
            output = Path(plan["output"])
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
            sources = [Path(path) for path in plan["source_shards"]]
            missing = [str(path) for path in sources if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing source shard(s) for {plan['noise_level']}: {missing}")
            initialize_adapter(
                sources, output, seed=args.seed, in_dim=48,
                model_dim=spec["dim"], downscale_factor=spec["downscale_factor"],
            )
            write_initializer_metadata(
                output, plan["noise_level"],
                model_type=spec["model_type"],
                initialization_seed=args.seed, source_shards=[path.name for path in sources],
            )
            print(f"Created {output}")
    except (ConfigError, FileNotFoundError, FileExistsError) as exc:
        parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

