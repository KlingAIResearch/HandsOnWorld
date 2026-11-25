"""Build inspectable torchrun commands without loading any neural networks."""

from __future__ import annotations

from pathlib import Path
import sys

from handsonworld.training.config import ConfigError
from .config import latest_checkpoint


def resolve_stage_inputs(config, *, dry_run=False):
    options = config["distillation"]
    stage = options["stage"]
    root = Path(options["output_root"])
    if not options.get("resume") and not options.get("student_init"):
        predecessor = {"ode": "causal_sft", "kd": "causal_sft", "dmd": "kd"}.get(stage)
        if predecessor:
            options["student_init"] = latest_checkpoint(root / predecessor, required=not dry_run)
    if stage == "infer":
        inference = config.setdefault("inference", {})
        if not inference.get("checkpoint"):
            inference["checkpoint"] = latest_checkpoint(root / "dmd", required=not dry_run)
        if not inference.get("output_dir") and not inference.get("output"):
            inference["output_dir"] = str(root / "inference")
    return config


def training_arguments(config):
    options = config["distillation"]
    stage = options["stage"]
    if stage not in ("causal_sft", "kd", "dmd"):
        raise ConfigError(f"Not a training stage: {stage}")
    shards = config["models"]["single"].get("source_shards", [])
    if not shards or len({str(Path(path).parent) for path in shards}) != 1:
        raise ConfigError("Distillation source_shards must reside in one original Wan directory")
    args = ["--stage", "sft" if stage == "causal_sft" else stage,
            "--model-type", config["model_type"],
            "--weights", str(Path(shards[0]).parent),
            "--control-checkpoint", options["teacher_checkpoint"],
            "--control-representation", config["representation"],
            "--diffsynth-root", config["backend_root"],
            "--data", options["ode_dir"] if stage == "kd" else options["cache_dir"],
            "--output", str(Path(options["output_root"]) / stage),
            "--height", str(config["training"]["height"]),
            "--width", str(config["training"]["width"]),
            "--video-frames", str(config["training"].get("num_frames", 81))]
    for key in ("chunk_size", "steps", "schedule_shift", "iterations", "save_every", "storage_dtype",
                "sharding_strategy", "guidance_scale", "student_update_freq", "lr", "fake_score_lr",
                "num_workers", "seed", "ode_substeps"):
        args.extend(("--" + key.replace("_", "-"), str(options[key])))
    for key in ("save_optimizer", "fixed_first_frame"):
        args.append("--" + ("" if options[key] else "no-") + key.replace("_", "-"))
    # Output reservation is an optional operational policy, not an algorithm requirement.
    args.append("--preallocate-checkpoints" if options.get("preallocate_checkpoints", False)
                else "--no-preallocate-checkpoints")
    for key in ("student_init", "resume"):
        if options.get(key):
            args.extend(("--" + key.replace("_", "-"), options[key]))
    if stage == "dmd" and options["streaming_student_backward"]:
        args.append("--streaming-student-backward")
    if stage == "kd" and options["allow_cross_control_ode"]:
        args.append("--allow-cross-control-ode")
    for key in ("t_list", "save_steps", "adam_betas"):
        if options.get(key) is not None:
            args.extend(("--" + key.replace("_", "-"), *(str(v) for v in options[key])))
    return args


def launch_command(config, config_path, topology=None, *, python=None):
    command = [python or sys.executable, "-m", "torch.distributed.run"]
    if topology is None:
        command += ["--standalone", "--nproc_per_node", "1"]
    else:
        machines, rank, processes, address, port = topology
        if (machines < 1 or not 0 <= rank < machines or processes < machines
                or processes % machines or not address or not 1 <= port <= 65535):
            raise ConfigError("Invalid torchrun node/process topology")
        command += ["--nnodes", str(machines), "--node_rank", str(rank),
                    "--nproc_per_node", str(processes // machines),
                    "--master_addr", address, "--master_port", str(port)]
    stage = config["distillation"]["stage"]
    if stage in ("causal_sft", "kd", "dmd"):
        command += ["--module", "handsonworld.distillation.train", *training_arguments(config)]
    else:
        command += ["--module", "handsonworld.distillation.worker", "--stage", stage,
                    "--config", str(Path(config_path).resolve())]
    return command
