#!/usr/bin/env python3
"""Launch cached causal SFT, ODE, KD, Self-Forcing and causal inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handsonworld.distillation.bootstrap import worker_environment  # noqa: E402
from handsonworld.distillation.commands import launch_command, resolve_stage_inputs  # noqa: E402
from handsonworld.distillation.config import STAGES, load_distill_config, stage_config  # noqa: E402
from handsonworld.training.config import ConfigError  # noqa: E402
from handsonworld.training.distributed import NodeCoordinator  # noqa: E402
from handsonworld.training.metadata import validate_weight_metadata  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--stage", choices=STAGES, required=True)
    result.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--sharding-strategy", choices=("node", "full"))
    result.add_argument("--student-init")
    result.add_argument("--resume")
    result.add_argument("--iterations", type=int)
    result.add_argument("--num-machines", type=int)
    result.add_argument("--machine-rank", type=int)
    result.add_argument("--num-processes", type=int)
    result.add_argument("--main-process-ip")
    result.add_argument("--main-process-port", type=int, default=29500)
    result.add_argument("--launch-id")
    result.add_argument("--rendezvous-dir", type=Path)
    result.add_argument("--rendezvous-timeout", type=float, default=600)
    result.add_argument("--offline", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def prepare_assets(config, path, offline):
    from handsonworld.training.resources import prepare_resources, link_resource
    from handsonworld.models.specs import get_model_spec

    options = config["distillation"]
    stage = options["stage"]
    inference = config.get("inference", {})
    marker = Path(inference.get("checkpoint", "")) / "complete.json"
    bundle = (stage == "infer" and marker.is_file()
              and json.loads(marker.read_text()).get("format") == "handsonworld_causal_student_v1")
    # Exported students already carry their adapter; no original LoRA is needed.
    if not bundle and stage != "prepare":
        validate_weight_metadata(options["teacher_checkpoint"], config["representation"],
                                 "single", model_type=config["model_type"])
    asset_keys = None if stage == "prepare" else {"faces"}
    if stage == "infer":
        asset_keys = None if inference.get("metadata") else {"faces", "vae"}
        if inference.get("metadata"):
            config = copy.deepcopy(config)
            config["data"]["metadata"] = inference["metadata"]
    prepare_resources(config, path, levels=(), offline=offline, reuse_roots=[ROOT], asset_keys=asset_keys)
    if stage not in ("causal_sft", "kd", "dmd"):
        return
    spec = get_model_spec(config["model_type"])
    for filename in config["models"]["single"]["source_shards"]:
        target = Path(filename)
        if target.is_file():
            continue
        if offline:
            raise FileNotFoundError(f"Missing original Wan shard: {target}")
        from huggingface_hub import hf_hub_download
        source = hf_hub_download(repo_id=spec["model_id"], filename=target.name,
                                 revision=config.get("resources", {}).get("revision", "main"))
        link_resource(Path(source), target)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        config = stage_config(load_distill_config(args.config), args.stage, {
            "save_optimizer": args.save_optimizer, "sharding_strategy": args.sharding_strategy,
            "student_init": str(Path(args.student_init).expanduser().resolve()) if args.student_init else None,
            "resume": str(Path(args.resume).expanduser().resolve()) if args.resume else None,
            "iterations": args.iterations,
        })
        config = resolve_stage_inputs(config, dry_run=args.dry_run)
        topology = None
        if args.num_machines is not None:
            if args.machine_rank is None or args.num_processes is None or not args.main_process_ip:
                raise ConfigError("Explicit cluster topology requires rank, process count and master address")
            topology = (args.num_machines, args.machine_rank, args.num_processes,
                        args.main_process_ip, args.main_process_port)
        elif any(v is not None for v in (args.machine_rank, args.num_processes, args.main_process_ip)):
            raise ConfigError("Cluster arguments require num-machines")
        command = launch_command(config, args.config, topology)
        print(f"[{args.stage}] " + shlex.join(command), flush=True)
        if args.dry_run:
            print(f"model={config['model_type']} representation={config['representation']} "
                  f"sharding={config['distillation']['sharding_strategy']} "
                  f"save_optimizer={config['distillation']['save_optimizer']}")
            return 0
        coordinator = None
        if args.num_machines and args.num_machines > 1:
            if not args.launch_id or not args.rendezvous_dir or args.rendezvous_timeout <= 0:
                raise ConfigError("Multi-node launch requires unique launch-id, rendezvous-dir and positive timeout")
            contract = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            coordinator = NodeCoordinator(args.rendezvous_dir, args.launch_id, args.machine_rank,
                                          args.num_machines, args.rendezvous_timeout, contract)

        def assets():
            if not args.machine_rank:
                prepare_assets(config, args.config, args.offline)

        if coordinator:
            coordinator.run("distillation-assets", assets)
        else:
            assets()
        env = worker_environment(config)
        # The resolved config includes CLI overrides and exact predecessor IDs.
        # Pass it through the environment to avoid rank races writing a config file.
        env["HANDSONWORLD_DISTILL_CONFIG"] = json.dumps(config, sort_keys=True)
        def run():
            subprocess.run(command, env=env, check=True)
        if coordinator:
            coordinator.run(f"distillation-{args.stage}", run)
        else:
            run()
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
