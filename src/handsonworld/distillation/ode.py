"""Generate causal teacher ODE paths with complete provenance."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from handsonworld.data.cache import CacheDataset, atomic_json, sha256_file, validate_timesteps
from handsonworld.data.trajectories import ODE_FORMAT, ODEDataset
from .schedule import euler_sigmas, student_sigmas


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


@torch.no_grad()
def generate_trajectory(model, data: dict[str, Any], args, *, reuse_kv_cache: bool = False):
    """Port of the proven chunkwise causal Euler path generator.

    The model wrapper must expose ``net`` and ``_prepare_training_data``; the
    shared causal network owns the KV implementation and noise scheduler.
    """
    net = model.net
    net.eval().requires_grad_(False)
    _, positive, negative = model._prepare_training_data(data)
    real = data["real"]
    device = real.device
    boundaries = validate_timesteps(args.t_list, "t_list")
    dense = euler_sigmas(boundaries, args.ode_substeps, args.schedule_shift)

    def clear():
        net.clear_caches(reuse_storage=True) if reuse_kv_cache else net.clear_caches()

    clear()
    stored = torch.empty((1, len(boundaries) - 1, *real.shape[1:]), dtype=torch.bfloat16, device="cpu")
    final = torch.empty_like(real, device="cpu", memory_format=torch.contiguous_format)
    latent_frames = real.shape[2]
    ranges = []
    start = 0
    if args.fixed_first_frame:
        prefix = positive["first_frame_cond"][:, :, :1].to(real)
        final[:, :, :1] = prefix.cpu()
        stored[:, :, :, :1] = prefix.cpu().unsqueeze(1)
        for tag, condition in (("pos", positive), ("neg", negative)):
            if tag == "neg" and args.guidance_scale == 1:
                continue
            net(prefix, torch.zeros(1, device=device, dtype=torch.float64), condition=condition,
                cache_tag=tag, cur_start_frame=0, store_kv=True, is_ar=True)
        start = 1
    while start < latent_frames:
        end = min(latent_frames, start + args.chunk_size)
        ranges.append((start, end))
        start = end
    for start, end in ranges:
        noise = torch.randn_like(real[:, :, start:end])
        initial = torch.tensor(boundaries[0], device=device, dtype=torch.float64)
        x = net.noise_scheduler.latents(noise, t_init=initial)
        for step in range(len(boundaries) - 1):
            stored[:, step, :, start:end] = x.cpu()
            for substep in range(args.ode_substeps):
                offset = step * args.ode_substeps + substep
                t_value, next_value = dense[offset], dense[offset + 1]
                t = torch.full((1,), t_value, device=device, dtype=torch.float64)
                kwargs = dict(cur_start_frame=start, store_kv=False, is_ar=True, fwd_pred_type="flow")
                flow = net(x, t, condition=positive, cache_tag="pos", **kwargs)
                if args.guidance_scale != 1:
                    neg_flow = net(x, t, condition=negative, cache_tag="neg", **kwargs)
                    flow = neg_flow + args.guidance_scale * (flow - neg_flow)
                x = (x.float() + (next_value - t_value) * flow.float()).to(x.dtype)
        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite ODE endpoint")
        final[:, :, start:end] = x.cpu()
        for tag, condition in (("pos", positive), ("neg", negative)):
            if tag == "neg" and args.guidance_scale == 1:
                continue
            net(x, torch.zeros(1, device=device, dtype=torch.float64), condition=condition,
                cache_tag=tag, cur_start_frame=start, store_kv=True, is_ar=True)
    clear()
    return {
        "real": final[0].contiguous(),
        "path": stored[0].contiguous(),
        "path_timesteps": torch.tensor(boundaries[:-1], dtype=torch.float64),
    }


def _checkpoint_digest(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    marker = path / "complete.json"
    if marker.is_file():
        return sha256_file(marker)
    payload = "\n".join(
        f"{item.relative_to(path)}:{item.stat().st_size}"
        for item in sorted(path.rglob("*")) if item.is_file()
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_ode(
    config: dict[str, Any], config_path: str | Path,
    *, model=None, model_factory: Callable[[dict[str, Any], str | Path], Any] | None = None,
    trajectory_fn: Callable[..., dict[str, torch.Tensor]] = generate_trajectory,
) -> Path:
    settings = config["distillation"]
    cache_root = Path(settings["cache_dir"]).expanduser().resolve()
    output = Path(settings["ode_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    source_representation = config["representation"]
    ode_settings = {**settings, **settings.get("ode", {})}
    target_representation = ode_settings.get("target_control_representation", source_representation)
    supported_representations = {"normal_plucker", "reflection_plucker"}
    if source_representation not in supported_representations or target_representation not in supported_representations:
        raise ValueError("ODE source and target controls must be normal_plucker or reflection_plucker")
    allow_cross = bool(ode_settings.get("allow_cross_control", ode_settings.get("allow_cross_control_ode", False)))
    if source_representation != target_representation and not allow_cross:
        raise ValueError("Cross-control ODE use requires distillation.ode.allow_cross_control=true")
    dataset = CacheDataset(cache_root, config["model_type"], expected_control_representation=source_representation)
    steps = int(settings.get("steps", 4))
    shift = float(settings.get("schedule_shift", 5.0))
    boundaries = tuple(settings.get("t_list") or student_sigmas(steps, shift))
    boundaries = validate_timesteps(boundaries, "distillation.t_list")
    if len(boundaries) != steps + 1:
        raise ValueError("distillation.t_list must contain steps + 1 values")
    teacher = Path(settings.get("student_init") or settings["teacher_checkpoint"]).expanduser().resolve()
    identity = {
        "model_type": config["model_type"], "cache": str(cache_root),
        "cache_manifest_sha256": sha256_file(cache_root / "index.json"),
        "source_student_checkpoint": str(teacher), "source_student_checkpoint_digest": _checkpoint_digest(teacher),
        "control_representation": source_representation,
        "target_control_representation": target_representation, "allow_cross_control": allow_cross,
        "path_timesteps": list(boundaries[:-1]), "steps": steps,
        "schedule_shift": shift, "ode_substeps": int(ode_settings.get("substeps", ode_settings.get("ode_substeps", 12))),
        "guidance_scale": float(settings.get("guidance_scale", 1.0)),
        "chunk_size": int(settings.get("chunk_size", 4)),
        "fixed_first_frame": bool(settings.get("fixed_first_frame", True)),
    }
    generation_path = output / "generation_config.json"
    if rank == 0:
        if generation_path.exists() and json.loads(generation_path.read_text()) != identity:
            raise ValueError("Existing ODE generation settings differ; use another ode_dir")
        atomic_json(generation_path, identity)
    else:
        deadline = time.monotonic() + float(ode_settings.get("coordination_timeout", 1800))
        while not generation_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for rank zero to initialize ODE output")
            time.sleep(0.2)
        if json.loads(generation_path.read_text()) != identity:
            raise ValueError("ODE generation settings differ between ranks")
    if model is None:
        if model_factory is None:
            try:
                from handsonworld.distillation.train import build_model_for_stage as model_factory
            except (ImportError, AttributeError) as error:
                raise RuntimeError("No distillation model factory is available for ODE generation") from error
        model = model_factory(config, config_path)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    collective_forward = bool(getattr(model, "requires_collective_forward", False))
    if distributed:
        if rank == 0:
            for peer in range(world):
                marker = output / f".rank-{peer:05d}.done"
                if marker.exists():
                    marker.unlink()
        torch.distributed.barrier()
    device = torch.device(ode_settings.get("device", "cuda"))
    dtype = torch.bfloat16 if ode_settings.get("dtype", "bfloat16") == "bfloat16" else torch.float32
    args = SimpleNamespace(
        t_list=boundaries, ode_substeps=identity["ode_substeps"], schedule_shift=shift,
        guidance_scale=identity["guidance_scale"], chunk_size=identity["chunk_size"],
        fixed_first_frame=identity["fixed_first_frame"],
    )
    maximum = int(ode_settings.get("max_samples", 0))
    total = len(dataset) if maximum <= 0 else min(len(dataset), maximum)
    base_seed = int(ode_settings.get("seed", 123))
    records = []
    from safetensors.torch import save_file
    positions = (
        list(range(rank, total, world))
        if not collective_forward
        else [round_index * world + rank for round_index in range((total + world - 1) // world)]
    )
    for index in positions:
        source_index = index if index < total else index % total
        destination = output / f"trajectory-{index:08d}.safetensors" if index < total else None
        seed = base_seed + source_index
        if destination is not None and destination.is_file() and not collective_forward:
            records.append({"file": destination.name, "cache_index": index, "seed": seed})
            continue
        # Collective model wrappers require every rank to execute every round,
        # including padding ranks and partial-resume files that already exist.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        data = _batch(dataset[source_index], device, dtype)
        if hasattr(model, "prepare_inference_data"):
            data = model.prepare_inference_data(data)
        result = trajectory_fn(model, data, args, reuse_kv_cache=bool(ode_settings.get("reuse_kv_cache", False)))
        if destination is not None and not destination.is_file():
            temporary = destination.with_name(destination.stem + ".tmp.safetensors")
            save_file(result, str(temporary))
            os.replace(temporary, destination)
        if destination is not None:
            records.append({"file": destination.name, "cache_index": index, "seed": seed})
    done = output / f".rank-{rank:05d}.done"
    done.write_text(json.dumps({"rank": rank, "world": world}) + "\n", encoding="utf-8")
    if distributed:
        torch.distributed.barrier()
    if rank != 0:
        return output / "index.json"
    if not distributed:
        deadline = time.monotonic() + float(ode_settings.get("coordination_timeout", 1800))
        while not all((output / f".rank-{peer:05d}.done").is_file() for peer in range(world)):
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for ODE worker ranks")
            time.sleep(0.5)
    records = [
        {"file": f"trajectory-{index:08d}.safetensors", "cache_index": index, "seed": base_seed + index}
        for index in range(total)
    ]
    manifest = {
        "format": ODE_FORMAT, "schema_version": 1, "complete": True,
        **identity, "samples": records,
        "provenance": {
            "solver": "euler", "generator": "handsonworld.distillation.ode",
            "model_type": config["model_type"],
            "fixed_first_frame": identity["fixed_first_frame"],
            "chunk_size": identity["chunk_size"],
            "schedule_shift": identity["schedule_shift"],
            "path_timesteps": identity["path_timesteps"],
            "source_student_checkpoint_digest": identity["source_student_checkpoint_digest"],
        },
    }
    atomic_json(output / "index.json", manifest)
    ODEDataset(
        output / "index.json", config["model_type"], cache_dir=cache_root,
        expected_path_timesteps=boundaries[:-1], expected_control_representation=target_representation,
    )
    return output / "index.json"
