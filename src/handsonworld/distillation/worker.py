"""Non-training stages executed by the shared torchrun launcher."""

import argparse
import json
import os

from .bootstrap import configure_backends
from .commands import resolve_stage_inputs
from .config import load_distill_config, stage_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "ode", "infer"), required=True)
    args = parser.parse_args()
    supplied = os.environ.get("HANDSONWORLD_DISTILL_CONFIG")
    config = json.loads(supplied) if supplied else resolve_stage_inputs(
        stage_config(load_distill_config(args.config), args.stage))
    if config["distillation"]["stage"] != args.stage:
        raise ValueError("Resolved configuration stage does not match worker stage")
    configure_backends(config)
    if args.stage == "prepare":
        from .prepare import prepare
        prepare(config, args.config)
    elif args.stage == "ode":
        from .ode import generate_ode
        generate_ode(config, args.config)
    else:
        from .infer import run_inference
        run_inference(config, args.config)


if __name__ == "__main__":
    main()
