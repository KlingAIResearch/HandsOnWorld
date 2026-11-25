"""Causal student inference from a finished checkpoint and public metadata."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from handsonworld.data.cache import CacheDataset, atomic_json, sha256_file
from handsonworld.training.config import resolve_path


def _resolve_checkpoint(config: dict[str, Any]) -> Path:
    inference = config.get("inference", {})
    value = inference.get("checkpoint") or config["distillation"].get("student_init")
    if not value:
        raise ValueError("Set inference.checkpoint to a finished student checkpoint")
    path = Path(value).expanduser().resolve()
    marker = path / "complete.json" if path.is_dir() else path.with_suffix(path.suffix + ".json")
    if path.is_dir() and not marker.is_file():
        raise ValueError(f"Student checkpoint is incomplete: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _default_sample(model, data, config):
    settings = config["distillation"]
    if hasattr(model, "sample_data"):
        return model.sample_data(
            data, steps=int(settings.get("steps", 4)), t_list=settings.get("t_list"),
            schedule_shift=float(settings.get("schedule_shift", 5.0)),
        )
    if hasattr(model, "prepare_inference_data"):
        data = model.prepare_inference_data(data)
    if not hasattr(model, "_prepare_training_data") or not hasattr(model, "net"):
        raise TypeError("Inference model must expose net and _prepare_training_data")
    _, positive, _ = model._prepare_training_data(data)
    noise = torch.randn_like(data["real"])
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    return CausVidModel.generator_fn(
        net=model.net, noise=noise, condition=positive,
        student_sample_steps=int(settings.get("steps", 4)),
        t_list=settings.get("t_list"), student_sample_type="sde", precision_amp=None,
    )


def _batch(sample: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    result = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(0)
            if value.is_floating_point():
                target_dtype = torch.float32 if key in {
                    "raw_cam_pose", "raw_hand_pose", "raw_camera_intrinsics",
                } else dtype
                value = value.to(device=device, dtype=target_dtype)
            else:
                value = value.to(device=device)
        result[key] = value
    return result


def _prepare_inference_cache(
    config: dict[str, Any], config_path: str | Path, output: Path,
    *, prepare_fn=None,
) -> tuple[dict[str, Any], Path]:
    """Encode inference metadata before any student model is materialized."""
    inference = config.get("inference", {})
    prepared = copy.deepcopy(config)
    prepared["data"]["metadata"] = inference["metadata"]
    if inference.get("data_root"):
        prepared["data"]["base_path"] = inference["data_root"]
    else:
        prepared["data"]["base_path"] = str(
            resolve_path(config_path, inference["metadata"]).parent
        )
    cache_root = Path(inference.get("cache_dir", output / ".condition_cache")).expanduser().resolve()
    prepared["distillation"]["cache_dir"] = str(cache_root)
    prepare_options = dict(prepared["distillation"].get("prepare", {}))
    for source, target in (
        ("prepare_device", "device"), ("prepare_dtype", "dtype"),
        ("max_samples", "max_samples"),
    ):
        if inference.get(source) is not None:
            prepare_options[target] = inference[source]
    prepared["distillation"]["prepare"] = prepare_options
    if prepare_fn is None:
        from .prepare import prepare as prepare_fn
    index_path = prepare_fn(prepared, config_path)
    rank = int(os.environ.get("RANK", 0))
    signature = hashlib.sha256(json.dumps({
        "model_type": prepared["model_type"], "data": prepared["data"],
        "training": prepared["training"], "cache_dir": str(cache_root),
        "prepare": prepare_options,
    }, sort_keys=True).encode()).hexdigest()
    ready_path = cache_root / f".inference-ready-{signature}.json"
    if rank == 0:
        validated = CacheDataset(
            cache_root, prepared["model_type"],
            expected_control_representation=prepared["representation"],
        )
        atomic_json(ready_path, {
            "signature": signature, "manifest_sha256": validated.manifest_sha256,
        })
    deadline = time.monotonic() + float(inference.get("prepare_timeout", 24 * 60 * 60))
    while True:
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            manifest = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ready = manifest = None
        if (
            isinstance(manifest, dict) and manifest.get("complete") is True
            and isinstance(ready, dict) and ready.get("signature") == signature
            and ready.get("manifest_sha256") == sha256_file(index_path)
        ):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Rank {rank} timed out waiting for inference cache {index_path}")
        time.sleep(0.25)
    # Drop T5/CLIP/VAE objects and allocator caches before loading the DiT.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prepared, cache_root


def _output_path(output: Path, sample: dict[str, Any], position: int, frames: int, layout: str) -> Path:
    suffix = f"_s0000_e{frames - 1:04d}.mp4"
    if layout == "diffsynth":
        source = sample.get("source_reference") or {}
        video = source.get("video") if isinstance(source, dict) else None
        if not video:
            raise ValueError("diffsynth output layout requires source video provenance in the prepared cache")
        root = Path(sample.get("data_root") or "/")
        try:
            relative = Path(video).resolve().relative_to(root.resolve()).with_suffix("")
        except ValueError as error:
            raise ValueError(f"Source video is outside inference data_root: {video}") from error
        destination = (output / relative).with_name(relative.name + suffix)
    elif layout == "flat":
        sample_id = str(sample["sample_id"])
        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in sample_id)[:120]
        destination = output / f"{position:06d}_{safe}{suffix}"
    else:
        raise ValueError("inference.output_layout must be flat or diffsynth")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _default_decode(records, config, config_path, output):
    inference = config.get("inference", {})
    backend = Path(config["backend_root"]).expanduser().resolve()
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.wan_video import WanVideoPipeline
    from diffsynth.utils.data import save_video
    from handsonworld.training.config import resolve_path

    device = torch.device(inference.get("device", "cuda"))
    dtype = torch.bfloat16 if inference.get("dtype", "bfloat16") == "bfloat16" else torch.float32
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=[ModelConfig(path=str(resolve_path(config_path, config["models"]["vae"])))],
        tokenizer_config=None, audio_processor_config=None, redirect_common_files=False,
    )
    for record in records:
        latent = record.pop("latent").to(device=device, dtype=dtype)
        decoded = pipe.vae.decode(latent, device=device, tiled=bool(inference.get("vae_tiled", True)))[0]
        frames = pipe.vae_output_to_video(decoded)
        save_video(frames, str(record["path"]), fps=int(inference.get("fps", 15)), quality=5)


def run_inference(
    config: dict[str, Any], config_path: str | Path,
    *, model=None, model_factory: Callable[..., Any] | None = None,
    sample_fn: Callable[[Any, dict[str, Any], dict[str, Any]], torch.Tensor] = _default_sample,
    decode_fn: Callable[[torch.Tensor, dict[str, Any], Path], None] | None = None,
    prepare_fn=None,
) -> Path:
    """Run sharded inference without any dependency on a training LMDB.

    Source metadata is first converted by the prepare stage, so large text,
    image and VAE encoders never coexist with the student DiT. The factory and
    decoder hooks keep the orchestration testable without CUDA.
    """
    settings = config["distillation"]
    inference = config.get("inference", {})
    checkpoint = _resolve_checkpoint(config)
    output = Path(
        inference.get("output_dir")
        or inference.get("output")
        or Path(settings["output_root"]) / "inference"
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if not 0 <= rank < world:
        raise ValueError("Inference rank must be smaller than world size")
    active_config = config
    cache_root = Path(settings["cache_dir"]).expanduser().resolve()
    if inference.get("metadata"):
        active_config, cache_root = _prepare_inference_cache(
            config, config_path, output, prepare_fn=prepare_fn,
        )
    cache = CacheDataset(
        cache_root, active_config["model_type"],
        expected_control_representation=active_config["representation"],
        include_metadata=True,
    )
    maximum = int(inference.get("max_samples", 0))
    total = len(cache) if maximum <= 0 else min(len(cache), maximum)
    if model is None:
        if model_factory is None:
            try:
                from handsonworld.distillation.train import build_inference_model as model_factory
            except (ImportError, AttributeError) as error:
                raise RuntimeError("No public inference model factory is available") from error
        model = model_factory(active_config, config_path, checkpoint=checkpoint)
    collective_forward = bool(getattr(model, "requires_collective_forward", False))
    device = torch.device(inference.get("device", "cuda"))
    dtype = torch.bfloat16 if inference.get("dtype", "bfloat16") == "bfloat16" else torch.float32
    records = []
    frames = int(active_config["training"].get("num_frames", 81))
    layout = inference.get("output_layout", "flat")
    data_root = Path(inference.get("data_root") or active_config["data"]["base_path"]).expanduser().resolve()
    positions = (
        list(range(rank, total, world))
        if not collective_forward
        else [round_index * world + rank for round_index in range((total + world - 1) // world)]
    )
    for position in positions:
        source_position = position if position < total else position % total
        sample = cache[source_position]
        sample_id = sample["sample_id"]
        sample["data_root"] = str(data_root)
        data = _batch(sample, device, dtype)
        seed = int(inference.get("seed", 0)) + source_position
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        latent = sample_fn(model, data, active_config)
        if isinstance(latent, (tuple, list)):
            latent = latent[0]
        if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or not torch.isfinite(latent).all():
            raise ValueError("Causal sampler must return a finite [B,C,T,H,W] latent tensor")
        if position >= total:
            continue
        destination = _output_path(output, sample, position, frames, layout)
        records.append({
            "position": position, "sample_id": sample_id, "seed": seed,
            "output": str(destination.relative_to(output)), "path": destination,
            "latent": latent.detach().cpu(),
        })
    # Do not keep the DiT and VAE resident at the same time.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not records:
        pass
    elif decode_fn is None:
        _default_decode(records, active_config, config_path, output)
    else:
        for record in records:
            decode_fn(record.pop("latent"), active_config, record["path"])
    public_records = [{key: value for key, value in record.items() if key != "path"} for record in records]
    marker = output / f"inference-rank{rank:04d}.json"
    atomic_json(marker, {
        "complete": True, "rank": rank, "world_size": world,
        "model_type": config["model_type"], "checkpoint": str(checkpoint), "samples": public_records,
    })
    return marker
