"""Configuration contract shared by the public HandsOnWorld entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPRESENTATIONS = ("normal_plucker", "reflection_plucker")
CONTROL_CHANNEL_ORDER = ("hand_plucker_world", "camera_plucker")
NOISE_RANGES = {
    "high": (0.0, 0.358),
    "low": (0.358, 1.0),
}
DEFAULT_MODEL_TYPE = "wan2.2-i2v-a14b"
MODEL_ALIASES = {
    "wan22_i2v14b": DEFAULT_MODEL_TYPE,
    "wan21_i2v14b": "wan2.1-i2v-14b-480p",
    "wan22_ti2v5b": "wan2.2-ti2v-5b",
}
MODEL_SPECS = {
    DEFAULT_MODEL_TYPE: {
        "model_id": "Wan-AI/Wan2.2-I2V-A14B", "dim": 5120,
        "downscale_factor": 8, "vae_filename": "Wan2.1_VAE.pth", "num_shards": 6,
        "image_encoder_filename": None, "levels": ("high", "low"),
        "initializer_filenames": {
            level: f"wan2.2_i2v14b_{level}_noise_dit_control_adapter_in_dim_48.safetensors"
            for level in ("high", "low")
        },
    },
    "wan2.1-i2v-14b-480p": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P", "dim": 5120,
        "downscale_factor": 8, "vae_filename": "Wan2.1_VAE.pth", "num_shards": 7,
        "image_encoder_filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "levels": ("single",),
        "initializer_filenames": {"single": "wan2.1_i2v14b_480p_dit_control_adapter_in_dim_48.safetensors"},
    },
    "wan2.2-ti2v-5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B", "dim": 3072,
        "downscale_factor": 16, "vae_filename": "Wan2.2_VAE.pth", "num_shards": 3,
        "image_encoder_filename": None, "levels": ("single",),
        "initializer_filenames": {"single": "wan2.2_5b_dit_control_adapter_in_dim_48.safetensors"},
    },
}


class ConfigError(ValueError):
    """Raised when a release configuration violates the paper contract."""


def model_spec(config: dict[str, Any]) -> dict[str, Any]:
    model_type = config.get("model_type", DEFAULT_MODEL_TYPE)
    if not isinstance(model_type, str) or model_type not in MODEL_SPECS:
        raise ConfigError(f"config.model_type must be one of {tuple(MODEL_SPECS)}, got {model_type!r}")
    return dict(MODEL_SPECS[model_type], model_type=model_type)


def model_levels(config: dict[str, Any]) -> tuple[str, ...]:
    return model_spec(config)["levels"]


def resolve_noise_levels(config: dict[str, Any], selection: str = "both") -> tuple[str, ...]:
    """The existing 'both' default means every expert required by the backbone."""
    levels = model_levels(config)
    if selection in ("auto", "both"):
        return levels
    if selection not in levels:
        raise ConfigError(f"--noise-level {selection} is incompatible with {model_spec(config)['model_type']}; use {levels}")
    return (selection,)


def _need(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] in (None, "", []):
        raise ConfigError(f"Missing required config value: {where}.{key}")
    return mapping[key]


def populate_data_root(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Default an omitted dataset root to the metadata file's directory."""
    data = config.get("data")
    if not isinstance(data, dict) or data.get("base_path") not in (None, ""):
        return config
    metadata = data.get("metadata")
    if isinstance(metadata, str) and metadata:
        data["base_path"] = str(resolve_path(config_path, metadata).parent)
    return config


def load_config(path: str | Path, purpose: str) -> dict[str, Any]:
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError("The config root must be a JSON object")
    populate_data_root(config, path)
    validate_config(config, purpose)
    return config


def validate_config(config: dict[str, Any], purpose: str) -> None:
    if purpose not in {"train", "infer", "initialize"}:
        raise ValueError(f"Unknown config purpose: {purpose}")
    spec = model_spec(config)

    representation = config.get("representation") if purpose == "initialize" else _need(config, "representation", "config")
    if representation not in REPRESENTATIONS and not (purpose == "initialize" and representation is None):
        raise ConfigError(
            f"config.representation must be one of {REPRESENTATIONS}, got {representation!r}"
        )
    _need(config, "backend_root", "config")
    if purpose != "initialize":
        _need(config, "hand_mesh_faces_path", "config")

    control = _need(config, "control", "config")
    if not isinstance(control, dict):
        raise ConfigError("config.control must be an object")
    if control.get("in_dim") != 48:
        raise ConfigError("config.control.in_dim must be 48")
    if control.get("attr_space") != "world":
        raise ConfigError("config.control.attr_space must be 'world'")
    channel_order = tuple(control.get("channel_order", ()))
    if channel_order != CONTROL_CHANNEL_ORDER:
        raise ConfigError(
            "config.control.channel_order must be " + repr(list(CONTROL_CHANNEL_ORDER))
        )

    models = _need(config, "models", "config")
    if not isinstance(models, dict):
        raise ConfigError("config.models must be an object")
    if purpose != "initialize":
        _need(models, "text_encoder", "config.models")
        _need(models, "tokenizer", "config.models")
        _need(models, "vae", "config.models")
        if spec["image_encoder_filename"]:
            _need(models, "image_encoder", "config.models")
    unexpected = set(models) & ({"high", "low", "single"} - set(spec["levels"]))
    if unexpected:
        raise ConfigError(f"config.models has incompatible experts for {spec['model_type']}: {sorted(unexpected)}")
    ranges = NOISE_RANGES if spec["levels"] == ("high", "low") else {"single": (0.0, 1.0)}
    for level, expected_range in ranges.items():
        model = _need(models, level, "config.models")
        if not isinstance(model, dict):
            raise ConfigError(f"config.models.{level} must be an object")
        _need(model, "adapter", f"config.models.{level}")
        actual_range = (model.get("min_timestep_boundary"), model.get("max_timestep_boundary"))
        if actual_range != expected_range:
            raise ConfigError(
                f"config.models.{level} timestep range must be {expected_range}, got {actual_range}"
            )
        if purpose == "initialize":
            shards = _need(model, "source_shards", f"config.models.{level}")
            if not isinstance(shards, list) or not all(isinstance(item, str) and item for item in shards):
                raise ConfigError(f"config.models.{level}.source_shards must be a non-empty string list")

    if purpose == "train":
        data = _need(config, "data", "config")
        training = _need(config, "training", "config")
        _need(data, "metadata", "config.data")
        if training.get("lora_rank") != 32:
            raise ConfigError("config.training.lora_rank must be 32")
        _need(training, "output_root", "config.training")
        if int(training.get("num_epochs", 0)) <= 0:
            raise ConfigError("config.training.num_epochs must be positive")
        if not isinstance(training.get("release_cache_before_backward", False), bool):
            raise ConfigError("config.training.release_cache_before_backward must be a boolean")
    elif purpose == "infer":
        inference = _need(config, "inference", "config")
        checkpoint_keys = ("high_checkpoint", "low_checkpoint") if len(spec["levels"]) == 2 else ("checkpoint",)
        for key in (
            *checkpoint_keys, "video", "cam_pose", "hand_pose",
            "prompt", "output", "original_width", "original_height",
        ):
            _need(inference, key, "config.inference")
        if len(spec["levels"]) == 2:
            switch = inference.get("switch_dit_boundary", 0.9)
            if not 0.0 < float(switch) < 1.0:
                raise ConfigError("config.inference.switch_dit_boundary must be between 0 and 1")


def resolve_path(config_path: str | Path, value: str) -> Path:
    """Resolve user paths relative to the config, never to a developer machine."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Preserve a configured symlink alias: its sidecar belongs beside the alias,
    # not beside the original read-only weight or Hugging Face cache entry.
    return Path(os.path.abspath(Path(config_path).resolve().parent / path))
