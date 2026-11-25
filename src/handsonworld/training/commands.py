"""Build reproducible DiffSynth commands without cluster-specific assumptions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import model_spec, resolve_noise_levels, resolve_path


def training_command(
    config: dict[str, Any], config_path: str | Path, noise_level: str,
    *, accelerate_config: str | Path | None = None,
    accelerate_args: list[str] | None = None,
) -> list[str]:
    resolve_noise_levels(config, noise_level)
    backend = resolve_path(config_path, config["backend_root"])
    model = config["models"][noise_level]
    data = config["data"]
    train = config["training"]
    output = resolve_path(config_path, train["output_root"]) / noise_level
    model_paths = [
        str(resolve_path(config_path, model["adapter"])),
        str(resolve_path(config_path, config["models"]["text_encoder"])),
        str(resolve_path(config_path, config["models"]["vae"])),
    ]
    if model_spec(config)["image_encoder_filename"]:
        model_paths.append(str(resolve_path(config_path, config["models"]["image_encoder"])))
    command = [
        sys.executable,
        str(backend / "examples/wanvideo/model_training/train.py"),
        "--dataset_base_path", str(resolve_path(config_path, data["base_path"])),
        "--dataset_metadata_path", str(resolve_path(config_path, data["metadata"])),
        "--data_file_keys", "video,cam_pose,hand_pose,camera_intrinsics",
        "--random_drop_keys", "prompt,cam_pose,hand_pose",
        "--height", str(train.get("height", 480)),
        "--width", str(train.get("width", 640)),
        "--num_frames", str(train.get("num_frames", 81)),
        "--hand_pose_render_mode", config["representation"],
        "--hand_pose_attr_space", "world",
        "--random_start_frame",
        "--dataset_repeat", "1",
        "--model_paths", json.dumps(model_paths),
        "--tokenizer_path", str(resolve_path(config_path, config["models"]["tokenizer"])),
        "--learning_rate", str(train.get("learning_rate", 1e-4)),
        "--num_epochs", str(train["num_epochs"]),
        "--remove_prefix_in_ckpt", "pipe.",
        "--output_path", str(output),
        "--trainable_models", "dit.control_adapter",
        "--lora_base_model", "dit",
        "--lora_target_modules", "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank", "32",
        "--extra_inputs", "input_image,cam_pose,hand_pose,camera_intrinsics",
        "--max_timestep_boundary", str(model["max_timestep_boundary"]),
        "--min_timestep_boundary", str(model["min_timestep_boundary"]),
        "--dataset_num_workers", str(train.get("dataset_num_workers", 16)),
    ]
    if train.get("use_gradient_checkpointing", False):
        command.append("--use_gradient_checkpointing")
    if train.get("release_cache_before_backward", False):
        command.append("--release_cache_before_backward")
    resume = train.get("resume", {}).get(noise_level) if isinstance(train.get("resume"), dict) else None
    if resume:
        command.extend(["--resume", str(resolve_path(config_path, resume))])
    if accelerate_args is not None:
        if accelerate_config is not None:
            raise ValueError("Cannot combine Accelerate config with explicit topology")
        command = [sys.executable, "-m", "accelerate.commands.launch", *accelerate_args, *command[1:]]
    elif accelerate_config is not None:
        command = [sys.executable, "-m", "accelerate.commands.launch",
                   "--config_file", str(Path(accelerate_config).expanduser().resolve()), *command[1:]]
    return command
