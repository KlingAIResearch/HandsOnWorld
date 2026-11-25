"""Sidecar metadata used to prevent incompatible control weights from loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import CONTROL_CHANNEL_ORDER, DEFAULT_MODEL_TYPE, ConfigError, model_spec


def expected_metadata(representation: str, noise_level: str,
                      model_type: str = DEFAULT_MODEL_TYPE) -> dict[str, Any]:
    spec = model_spec({"model_type": model_type})
    if noise_level not in spec["levels"]:
        raise ConfigError(f"Noise level {noise_level!r} is incompatible with {model_type}")
    payload = {
        "format": "handsonworld-control-v1",
        "representation": representation,
        "attr_space": "world",
        "control_in_dim": 48,
        "channel_order": list(CONTROL_CHANNEL_ORDER),
        "noise_level": noise_level,
    }
    # Keep existing A14B run manifests and sidecars compatible without rewriting.
    if model_type != DEFAULT_MODEL_TYPE:
        payload["model_type"] = model_type
    return payload


def metadata_path(weight_path: str | Path) -> Path:
    return Path(str(weight_path) + ".handsonworld.json")


def weight_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(weight_path: str | Path, representation: str, noise_level: str,
                   *, model_type: str = DEFAULT_MODEL_TYPE, **extra: Any) -> Path:
    output = metadata_path(weight_path)
    payload = expected_metadata(representation, noise_level, model_type)
    payload.update(extra)
    payload["weight_sha256"] = weight_digest(weight_path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def initializer_metadata(noise_level: str, model_type: str = DEFAULT_MODEL_TYPE) -> dict[str, Any]:
    spec = model_spec({"model_type": model_type})
    payload = expected_metadata("normal_plucker", noise_level, model_type)
    payload.pop("representation")
    payload.update(format="handsonworld-initializer-v1", model_type=model_type,
                   kind="zero_initialized_control_adapter", model_dim=spec["dim"],
                   downscale_factor=spec["downscale_factor"])
    return payload


def write_initializer_metadata(weight_path: str | Path, noise_level: str, *,
                               model_type: str = DEFAULT_MODEL_TYPE,
                               initialization_seed: int | None = None, **extra: Any) -> Path:
    """Write only after initialization or verification of a neutral imported model."""
    payload = dict(extra)
    payload.update(initializer_metadata(noise_level, model_type))
    payload["initialization_seed"] = initialization_seed
    payload["weight_sha256"] = weight_digest(weight_path)
    output = metadata_path(weight_path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_weight_metadata(
    weight_path: str | Path,
    representation: str,
    noise_level: str,
    *,
    required: bool = True,
    model_type: str = DEFAULT_MODEL_TYPE,
    allow_initializer: bool = False,
) -> dict[str, Any] | None:
    sidecar = metadata_path(weight_path)
    if not sidecar.is_file():
        if required:
            raise ConfigError(
                f"Missing weight metadata sidecar: {sidecar}. "
                "Create adapters with scripts/prepare_resources.py or publish matching metadata."
            )
        return None
    try:
        actual = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid weight metadata {sidecar}: {exc}") from exc
    if not isinstance(actual, dict):
        raise ConfigError(f"Weight metadata must be a JSON object: {sidecar}")
    if allow_initializer and actual.get("format") == "handsonworld-initializer-v1":
        expected = initializer_metadata(noise_level, model_type)
        if "representation" in actual:
            raise ConfigError(f"Shared initializer must not bind a representation: {sidecar}")
        if ("initialization_seed" not in actual or
                (actual["initialization_seed"] is not None and type(actual["initialization_seed"]) is not int)):
            raise ConfigError(f"Initializer initialization_seed must be an integer or null: {sidecar}")
    else:
        expected = expected_metadata(representation, noise_level, model_type)
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    actual_model_type = actual.get("model_type", DEFAULT_MODEL_TYPE)
    if actual_model_type != model_type:
        mismatches["model_type"] = (actual_model_type, model_type)
    if mismatches:
        details = ", ".join(
            f"{key}={got!r} (expected {want!r})" for key, (got, want) in mismatches.items()
        )
        raise ConfigError(f"Incompatible weight metadata for {weight_path}: {details}")
    if actual.get("weight_sha256") != weight_digest(weight_path):
        raise ConfigError(f"Weight SHA256 mismatch or missing hash: {weight_path}")
    if allow_initializer and actual.get("format") != "handsonworld-initializer-v1":
        # Legacy initializer metadata shared the trained-checkpoint schema.
        # Check the tensors before accepting that schema in a base-model slot.
        from .resources import verify_neutral_initializer
        verify_neutral_initializer(Path(weight_path), model_type=model_type)
    return actual


def write_run_metadata(output_dir: str | Path, config: dict[str, Any], noise_level: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "handsonworld-run.json"
    payload = expected_metadata(config["representation"], noise_level, model_spec(config)["model_type"])
    payload.update({
        "lora_rank": config["training"]["lora_rank"],
        "min_timestep_boundary": config["models"][noise_level]["min_timestep_boundary"],
        "max_timestep_boundary": config["models"][noise_level]["max_timestep_boundary"],
    })
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ConfigError(f"Output directory belongs to a different training configuration: {output_dir}")
    elif any(output_dir.glob("*.safetensors")):
        raise ConfigError(f"Output directory contains checkpoints without a run manifest: {output_dir}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path



def write_checkpoint_metadata(weight_path: str | Path) -> None:
    """Attach the run contract at each save, including before interrupted runs."""
    run_path = Path(weight_path).parent / "handsonworld-run.json"
    if not run_path.is_file():
        return
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    representation = payload.pop("representation")
    level = payload.pop("noise_level")
    write_metadata(weight_path, representation, level, **payload)
