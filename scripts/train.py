#!/usr/bin/env python3
"""Train HandsOnWorld control adapter + LoRA weights for the configured Wan backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from handsonworld.training.commands import training_command  # noqa: E402
from handsonworld.training.config import ConfigError, load_config, model_spec, resolve_noise_levels, resolve_path  # noqa: E402
from handsonworld.training.distributed import NodeCoordinator, cluster_arguments  # noqa: E402
from handsonworld.training.metadata import (  # noqa: E402
    validate_weight_metadata, write_run_metadata,
)
from handsonworld.training.resources import prepare_resources  # noqa: E402
from handsonworld.training.runtime import add_backend_to_path, validate_resources  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--noise-level", choices=("high", "low", "both", "single", "auto"), default="both",
                        help="Select an expert; both/auto runs all experts required by the configured model")
    result.add_argument("--accelerate-config", type=Path, help="Accelerate YAML launcher config; enables multi-GPU launch through this wrapper")
    result.add_argument("--num-machines", type=int)
    result.add_argument("--machine-rank", type=int)
    result.add_argument("--num-processes", type=int)
    result.add_argument("--main-process-ip")
    result.add_argument("--main-process-port", type=int, default=29500)
    result.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    result.add_argument("--launch-id")
    result.add_argument("--rendezvous-dir", type=Path)
    result.add_argument("--rendezvous-timeout", type=float, default=600)
    result.add_argument("--offline", action="store_true", help="Reuse local assets only; disable model downloads")
    result.add_argument("--no-prepare-resources", action="store_true", help="Only validate configured files; skip automatic resource preparation")
    result.add_argument("--reuse-root", type=Path, action="append", help="Local source root to search before downloading (repeatable)")
    result.add_argument("--resource-timeout", type=float, default=7200, help="Multi-node wait budget for asset preparation in seconds")
    result.add_argument("--prepare-only", action="store_true", help="Prepare and validate assets, then exit without starting training")
    result.add_argument("--dry-run", action="store_true", help="Validate config and print commands without weights")
    result.add_argument("--allow-missing-metadata", action="store_true", help="For legacy private adapters only")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, "train")
        levels = resolve_noise_levels(config, args.noise_level)
        model_type = model_spec(config)["model_type"]
        topology = cluster_arguments(args)
        if not math.isfinite(args.resource_timeout) or args.resource_timeout <= 0:
            raise ConfigError("resource-timeout must be finite and positive")
        commands = [training_command(config, args.config, level, accelerate_config=args.accelerate_config, accelerate_args=topology) for level in levels]
        for level, command in zip(levels, commands):
            print(f"[{level}] " + shlex.join(command))
        if args.dry_run:
            return 0
        rank = args.machine_rank or 0
        coordinator = None
        if args.num_machines is not None and args.num_machines > 1:
            contract = hashlib.sha256(json.dumps({
                "config": config, "levels": levels, "num_processes": args.num_processes,
                "num_machines": args.num_machines, "allow_missing_metadata": args.allow_missing_metadata,
                "config_directory": str(Path(args.config).resolve().parent),
                "offline": args.offline, "prepare_only": args.prepare_only,
                "no_prepare_resources": args.no_prepare_resources,
                "reuse_roots": [str(p.resolve()) for p in args.reuse_root] if args.reuse_root else None,
                "master": args.main_process_ip, "port": args.main_process_port,
                "precision": args.mixed_precision,
            }, sort_keys=True).encode()).hexdigest()
            coordinator = NodeCoordinator(args.rendezvous_dir, args.launch_id, rank,
                                          args.num_machines, args.rendezvous_timeout, contract)

        def stage(name, action, timeout=None):
            if coordinator is None:
                action()
            elif timeout is None:
                coordinator.run(name, action)
            else:
                coordinator.run(name, action, timeout=timeout)

        def check_resources():
            if args.accelerate_config is not None and not args.accelerate_config.expanduser().is_file():
                raise FileNotFoundError(f"Missing Accelerate launcher config: {args.accelerate_config}")
            add_backend_to_path(config, args.config)
            validate_resources(config, args.config, "train")

        if not args.no_prepare_resources:
            def prepare_assets():
                if rank == 0:
                    prepare_resources(config, args.config, levels=levels, offline=args.offline,
                                      reuse_roots=args.reuse_root)

            stage("prepare-assets", prepare_assets, timeout=args.resource_timeout)
        stage("resources", check_resources)
        backend = resolve_path(args.config, config["backend_root"])
        faces = resolve_path(args.config, config["hand_mesh_faces_path"])
        pythonpath = str(backend) + os.pathsep + str(REPO_ROOT / "src")
        if os.environ.get("PYTHONPATH"):
            pythonpath += os.pathsep + os.environ["PYTHONPATH"]
        env = dict(os.environ, HAND_MESH_FACES_PATH=str(faces), PYTHONPATH=pythonpath)
        for level, command in zip(levels, commands):
            output = resolve_path(args.config, config["training"]["output_root"]) / level

            def prepare():
                adapter = resolve_path(args.config, config["models"][level]["adapter"])
                if not adapter.is_file():
                    raise FileNotFoundError(f"Missing {level} adapter: {adapter}")
                print(f"[node {rank}] Checking {level} initializer metadata/SHA256: {adapter}", flush=True)
                validate_weight_metadata(adapter, config["representation"], level,
                                         required=not args.allow_missing_metadata, model_type=model_type, allow_initializer=True)
                for checkpoint in output.glob("*.safetensors"):
                    validate_weight_metadata(checkpoint, config["representation"], level, model_type=model_type)
                resume = config["training"].get("resume", {}).get(level)
                if resume:
                    validate_weight_metadata(resolve_path(args.config, resume), config["representation"], level,
                                             required=not args.allow_missing_metadata, model_type=model_type)
                if rank == 0 and not args.prepare_only:
                    write_run_metadata(output, config, level)

            stage(f"{level}-prepare", prepare)
            if args.prepare_only:
                continue
            # Only node 0 writes run metadata and audits output hashes. The barrier
            # keeps other nodes from reading output checkpoints while they are saved.
            def train_and_verify():
                subprocess.run(command, check=True, env=env)
                if rank == 0:
                    for checkpoint in output.glob("*.safetensors"):
                        validate_weight_metadata(checkpoint, config["representation"], level, model_type=model_type)

            stage(f"{level}-finished", train_and_verify)
        if args.prepare_only:
            print("Resources prepared and validated; training was not started.", flush=True)

    except (ConfigError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
