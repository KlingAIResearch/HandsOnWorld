"""Dataset planning for independent, per-GPU inference workers."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from pathlib import Path

from handsonworld.training.config import ConfigError, model_levels, resolve_path


def checkpoint_paths(config, config_path, high=None, low=None, epoch=None, checkpoint=None):
    """Honor explicit paths, then configured files, then the final training epoch."""
    levels = model_levels(config)
    if levels == ("single",):
        if high is not None or low is not None:
            raise ConfigError("Use --checkpoint for a single-DiT model")
        choices = (("single", checkpoint),)
    else:
        if checkpoint is not None:
            raise ConfigError("Use --high-checkpoint and --low-checkpoint for a dual-DiT model")
        choices = (("high", high), ("low", low))
    result = {}
    for level, explicit in choices:
        key = "checkpoint" if level == "single" else f"{level}_checkpoint"
        configured = resolve_path(config_path, config["inference"][key])
        if explicit:
            result[level] = str(Path(explicit).expanduser().resolve())
        elif epoch is None and configured.is_file():
            result[level] = str(configured)
        else:
            training = config.get("training", {})
            selected_epoch = epoch if epoch is not None else int(training.get("num_epochs", 2)) - 1
            if selected_epoch < 0 or not training.get("output_root"):
                raise ConfigError("Set checkpoint paths or a training output_root and valid epoch")
            result[level] = str(resolve_path(config_path, training["output_root"]) / level
                                / f"epoch-{selected_epoch}.safetensors")
    return result


def read_samples(metadata, data_root=None, split="all", max_samples=None, *, output_layout="flat"):
    if output_layout not in {"flat", "diffsynth"}:
        raise ConfigError("output_layout must be flat or diffsynth")
    if output_layout == "diffsynth" and data_root is None:
        raise ConfigError("diffsynth output layout requires --data-root")
    metadata = Path(metadata).expanduser().resolve()
    root = Path(data_root).expanduser().resolve() if data_root else metadata.parent
    with metadata.open(newline="", encoding="utf-8") as stream:
        rows = json.load(stream) if metadata.suffix.lower() == ".json" else list(csv.DictReader(stream))
    result = []
    seen_ids = set()
    for index, row in enumerate(rows):
        if row.get("status") not in (None, "", "success"):
            continue
        if split != "all" and row.get("split") != split:
            continue
        sample = dict(row)
        for key in ("video", "cam_pose", "hand_pose", "camera_intrinsics"):
            value = row.get(key)
            if not value:
                if key != "camera_intrinsics":
                    raise ConfigError(f"Metadata row {index + 1} is missing {key}")
                sample[key] = None
                continue
            path = Path(value).expanduser()
            sample[key] = str((root / path).resolve())
        if not isinstance(row.get("prompt"), str):
            raise ConfigError(f"Metadata row {index + 1} is missing prompt")
        # Only a validated relative source path may become a nested output path.
        if output_layout == "diffsynth":
            try:
                relative = Path(sample["video"]).relative_to(root)
            except ValueError as exc:
                raise ConfigError(f"Metadata row {index + 1} video is outside --data-root") from exc
            sample["sample_id"] = str(relative.with_suffix(""))
            if sample["sample_id"] in seen_ids:
                raise ConfigError(f"Duplicate output path in diffsynth layout: {relative}")
        else:
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(sample["video"]).stem)[:100]
            sample["sample_id"] = f"{index:06d}_{stem}"
        seen_ids.add(sample["sample_id"])
        result.append(sample)
        if max_samples is not None and len(result) >= max_samples:
            break
    return result


def shard_samples(samples, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ConfigError("Require world_size >= 1 and 0 <= rank < world_size")
    return samples[rank::world_size]


def clip_windows(total_frames, requested_frames, clips_per_video):
    if min(total_frames, requested_frames, clips_per_video) < 1:
        raise ConfigError("Frame counts and clips_per_video must be positive")
    length = min(total_frames, requested_frames)
    length -= (length - 1) % 4
    last = total_frames - length
    starts = [0] if clips_per_video == 1 else sorted({
        round(i * last / (clips_per_video - 1)) for i in range(clips_per_video)
    })
    return [(start, length) for start in starts]


def sample_config(config, sample, output_dir, start=0, length=None, comparison=True):
    result = copy.deepcopy(config)
    infer = result["inference"]
    for key in ("video", "cam_pose", "hand_pose", "camera_intrinsics", "prompt"):
        infer[key] = sample[key]
    for key, source in (("original_width", "width"), ("original_height", "height")):
        if sample.get(source):
            infer[key] = int(sample[source])
    infer["start_frame"] = start
    if length is not None:
        infer["num_frames"] = length
    end = start + int(infer.get("num_frames", 81)) - 1
    name = f"{sample['sample_id']}_s{start:04d}_e{end:04d}"
    infer["output"] = str(Path(output_dir).resolve() / f"{name}.mp4")
    infer.pop("comparison_output", None)
    if comparison:
        infer["comparison_output"] = str(Path(output_dir).resolve() / f"{name}_comparison.mp4")
    return result


def inference_code_digest(config, config_path):
    """Bind cached videos to the actual code, including local backend edits."""
    backend = resolve_path(config_path, config["backend_root"])
    manifest = backend / "backend_manifest.json"
    files = json.loads(manifest.read_text())["files"]
    paths = [manifest, *(backend / name for name in sorted(files)),
             Path(__file__), Path(__file__).parent / "training/runtime.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def job_signature(config, checkpoint_metadata, code_digest=None):
    payload = {"config": config, "checkpoint_metadata": checkpoint_metadata, "code_digest": code_digest}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def completed(config, signature):
    infer = config["inference"]
    output = Path(infer["output"])
    marker = output.with_suffix(".json")
    try:
        record = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    paths = [output]
    if infer.get("comparison_output"):
        paths.append(Path(infer["comparison_output"]))
    return record.get("signature") == signature and all(p.is_file() and p.stat().st_size > 0 for p in paths)
