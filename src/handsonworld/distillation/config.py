"""Public, CPU-only configuration for the two supported causal backbones."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from handsonworld.models.specs import DISTILLATION_MODEL_TYPES, get_model_spec
from handsonworld.training.config import ConfigError, populate_data_root, resolve_path, validate_config

STAGES = ("prepare", "causal_sft", "ode", "kd", "dmd", "infer")
DEFAULTS = {
    "chunk_size": 4, "fixed_first_frame": True, "steps": 4,
    "schedule_shift": 5.0, "iterations": 3000, "save_every": 1000,
    "save_optimizer": True, "storage_dtype": "float32", "guidance_scale": 1.0,
    "student_update_freq": 6, "lr": 2e-6, "fake_score_lr": 4e-7,
    "streaming_student_backward": True, "num_workers": 2, "seed": 123,
    "ode_substeps": 12, "allow_cross_control_ode": False,
}


def _read(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration must be an object: {path}")
    return value


def _normalize_paths(config, path):
    result = copy.deepcopy(config)
    for key in ("backend_root", "hand_mesh_faces_path"):
        if result.get(key):
            result[key] = str(resolve_path(path, result[key]))
    for key, value in result.get("models", {}).items():
        if isinstance(value, str):
            result["models"][key] = str(resolve_path(path, value))
        elif isinstance(value, dict):
            for name in ("adapter",):
                if value.get(name):
                    value[name] = str(resolve_path(path, value[name]))
            if "source_shards" in value:
                value["source_shards"] = [str(resolve_path(path, item)) for item in value["source_shards"]]
    for key in ("base_path", "metadata"):
        if result.get("data", {}).get(key):
            result["data"][key] = str(resolve_path(path, result["data"][key]))
    for key in ("output_root",):
        if result.get("training", {}).get(key):
            result["training"][key] = str(resolve_path(path, result["training"][key]))
    return result


def load_distill_config(path):
    """Load a recipe referencing an existing SFT config, resolving each origin.

    A recipe may override data/training fields; its relative paths belong to the
    recipe, while inherited paths continue to belong to the SFT configuration.
    """
    path = Path(path).expanduser().resolve()
    recipe = _read(path)
    base = recipe.get("base_config")
    if base:
        base_path = resolve_path(path, base)
        source = _read(base_path)
        if source.get("base_config"):
            raise ConfigError("base_config must reference a concrete SFT configuration")
        config = _normalize_paths(source, base_path)
        overrides = _normalize_paths({key: value for key, value in recipe.items()
                                      if key not in ("base_config", "distillation")}, path)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    else:
        config = _normalize_paths(recipe, path)
    populate_data_root(config, path)
    if config.get("model_type") not in DISTILLATION_MODEL_TYPES:
        raise ConfigError(f"Self-Forcing supports only {DISTILLATION_MODEL_TYPES}; A14B high/low is SFT-only")
    validate_config(config, "train")
    if not isinstance(recipe.get("distillation"), dict):
        raise ConfigError("Missing distillation configuration")
    options = {**DEFAULTS, **recipe["distillation"]}
    options.setdefault("sharding_strategy", "node" if config["model_type"] == "wan2.2-ti2v-5b" else "full")
    for key in ("teacher_checkpoint", "cache_dir", "ode_dir", "output_root"):
        if not isinstance(options.get(key), str) or not options[key]:
            raise ConfigError(f"distillation.{key} must be a path")
    for key in ("teacher_checkpoint", "cache_dir", "ode_dir", "output_root", "student_init", "resume"):
        if options.get(key):
            options[key] = str(resolve_path(path, options[key]))
    if not isinstance(options.get("stages", {}), dict):
        raise ConfigError("distillation.stages must be an object")
    for stage, value in options.get("stages", {}).items():
        if stage not in STAGES or not isinstance(value, dict):
            raise ConfigError(f"Invalid distillation stage override: {stage}")
        for key in ("student_init", "resume", "output", "checkpoint"):
            if value.get(key):
                value[key] = str(resolve_path(path, value[key]))
    # Student inference belongs to this recipe, not the referenced SFT model.
    # In particular, never inherit the teacher's checkpoint or output directory.
    legacy_inference = options.pop("inference", {})
    inference = recipe.get("inference", {})
    if not isinstance(legacy_inference, dict):
        raise ConfigError("distillation.inference must be an object")
    if not isinstance(inference, dict):
        raise ConfigError("inference must be an object")
    inference = {**legacy_inference, **inference}
    for key in ("checkpoint", "output", "output_dir", "metadata", "data_root", "cache_dir"):
        if inference.get(key):
            inference[key] = str(resolve_path(path, inference[key]))
    config["inference"] = inference
    config["distillation"] = options
    validate_distillation(config)
    return config


def stage_config(config, stage, overrides=None):
    if stage not in STAGES:
        raise ConfigError(f"Unknown distillation stage: {stage}")
    result = copy.deepcopy(config)
    settings = result["distillation"]
    settings.update(settings.get("stages", {}).get(stage, {}))
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    settings["stage"] = stage
    validate_distillation(result)
    return result


def validate_distillation(config):
    spec = get_model_spec(config["model_type"])
    opts = config["distillation"]
    for key in ("chunk_size", "steps", "iterations", "save_every", "student_update_freq", "ode_substeps"):
        if type(opts.get(key)) is not int or opts[key] <= 0:
            raise ConfigError(f"distillation.{key} must be a positive integer")
    for key in ("save_optimizer", "fixed_first_frame", "streaming_student_backward", "allow_cross_control_ode"):
        if not isinstance(opts.get(key), bool):
            raise ConfigError(f"distillation.{key} must be boolean")
    for key in ("lr", "fake_score_lr", "schedule_shift"):
        value = opts.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"distillation.{key} must be finite and positive")
    if opts["storage_dtype"] not in ("bfloat16", "float32"):
        raise ConfigError("storage_dtype must be bfloat16 or float32")
    if opts["sharding_strategy"] not in ("full", "node"):
        raise ConfigError("sharding_strategy must be full or node (hybrid sharding)")
    if opts.get("resume") and opts.get("student_init"):
        raise ConfigError("resume and student_init are mutually exclusive")
    shape = config["training"]
    frames = shape.get("num_frames", 81)
    if type(frames) is not int or frames < 5 or frames % 4 != 1:
        raise ConfigError("num_frames must be 4k+1 with at least one generated latent frame")
    for key in ("height", "width"):
        value = shape.get(key)
        multiple = spec["spatial_downscale"] * 2
        if type(value) is not int or value < multiple or value % multiple:
            raise ConfigError(f"{key} must be a positive multiple of {multiple}")
    times = opts.get("t_list")
    if times is not None:
        if (not isinstance(times, list) or len(times) != opts["steps"] + 1
                or any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times)
                or not math.isclose(times[0], 0.999, rel_tol=0, abs_tol=1e-9) or times[-1] != 0
                or any(a <= b for a, b in zip(times, times[1:]))):
            raise ConfigError("t_list must contain steps+1 decreasing sigmas from 0.999 to 0")


def latest_checkpoint(directory, *, required=True):
    """Select only committed checkpoint directories, never partial writes."""
    candidates = []
    for marker in Path(directory).glob("step-*/complete.json"):
        try:
            record = json.loads(marker.read_text())
            step = int(record["iteration"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        candidates.append((step, marker.parent))
    if candidates:
        return str(max(candidates)[1])
    if required:
        raise ConfigError(f"No completed checkpoint in {directory}; run the preceding stage or set student_init")
    return str(Path(directory) / "step-<latest-complete>")
