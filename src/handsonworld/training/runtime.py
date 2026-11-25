"""Lazy heavy-runtime functions for model initialization and inference."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from .config import model_levels, resolve_path


def add_backend_to_path(config: dict[str, Any], config_path: str | Path) -> Path:
    backend = resolve_path(config_path, config["backend_root"])
    if not (backend / "diffsynth").is_dir():
        raise FileNotFoundError(f"DiffSynth backend is missing: {backend / 'diffsynth'}")
    sys.path.insert(0, str(backend))
    return backend


def initialize_adapter(
    source_paths: list[Path], output_path: Path, *, seed: int = 0, in_dim: int = 48,
    model_dim: int = 5120, downscale_factor: int = 8,
) -> None:
    import torch
    from safetensors.torch import save_file
    from diffsynth.core.loader.file import load_state_dict
    from diffsynth.models.wan_video_camera_controller import SimpleAdapter

    torch.manual_seed(seed)
    state_dict = load_state_dict([str(path) for path in source_paths], device="cpu")
    dtype = next(value.dtype for value in state_dict.values() if isinstance(value, torch.Tensor))
    adapter = SimpleAdapter(
        in_dim, model_dim, kernel_size=(2, 2), stride=(2, 2),
        downscale_factor=downscale_factor,
    ).to(dtype=dtype)
    adapter_state = adapter.state_dict()
    adapter_state["conv.weight"].zero_()
    adapter_state["conv.bias"].zero_()
    for name, tensor in adapter_state.items():
        if name.endswith(".conv2.weight") or name.endswith(".conv2.bias"):
            tensor.zero_()
    state_dict.update({f"control_adapter.{key}": value for key, value in adapter_state.items()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace the configured alias, never truncate a linked original/cache file.
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(state_dict, str(temporary))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_lora_key(key: str) -> bool:
    return any(token in key for token in (
        ".lora_A.", ".lora_B.", ".lora_down.", ".lora_up.",
        ".lora_A.default.", ".lora_B.default.", "lora_A.weight", "lora_B.weight",
    ))


def _logical_module_key(key: str) -> str:
    """Remove transparent VRAM-management wrapper levels from a state key."""
    while key.startswith("module."):
        key = key[len("module."):]
    while ".module." in key:
        key = key.replace(".module.", ".")
    return key.removesuffix(".module")


def _logical_named_modules(module: Any) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name, child in module.named_modules():
        modules.setdefault(_logical_module_key(name), child)
        wrapper_name = getattr(child, "name", None)
        if isinstance(wrapper_name, str) and wrapper_name:
            modules.setdefault(wrapper_name, child)
    return modules


def load_finetuned_dit(pipe: Any, dit: Any, checkpoint_path: Path, label: str) -> None:
    """Load non-LoRA adapter state first, then attach LoRA to the same DiT."""
    import torch
    from diffsynth.core import load_state_dict

    state = load_state_dict(str(checkpoint_path), torch_dtype=pipe.torch_dtype, device="cpu")
    if any(key.startswith("pipe.") for key in state):
        state = {key[5:] if key.startswith("pipe.") else key: value for key, value in state.items()}
    dit_state = {key[4:]: value for key, value in state.items() if key.startswith("dit.")}
    if dit_state:
        state = dit_state
    lora = {key: value for key, value in state.items() if _is_lora_key(key)}
    adapter = {key: value for key, value in state.items() if not _is_lora_key(key)}
    if not lora:
        raise ValueError(f"{label} checkpoint contains no LoRA tensors: {checkpoint_path}")
    if dit is None or not hasattr(dit, "control_adapter"):
        raise ValueError(f"{label} DiT has no control adapter")
    raw_expected = dit.control_adapter.state_dict()
    expected = {
        "control_adapter." + _logical_module_key(key): value
        for key, value in raw_expected.items()
    }
    if len(expected) != len(raw_expected):
        raise ValueError(f"{label} DiT control adapter has ambiguous wrapped tensor names")
    if set(adapter) != set(expected):
        raise ValueError(f"{label} checkpoint has missing or unexpected control adapter tensors")
    for key, value in adapter.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"Invalid adapter tensor: {key}")
    # Validate every pair against a real target before the backend can silently skip it.
    normalized = {}
    for key, value in lora.items():
        key = key.replace(".default.", ".").replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
        if key in normalized or not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid or duplicated LoRA tensor: {key}")
        normalized[key] = value
    modules = _logical_named_modules(dit)
    used = set()
    for key, a in normalized.items():
        if not key.endswith(".lora_A.weight"):
            continue
        target = key.removesuffix(".lora_A.weight")
        b_key = target + ".lora_B.weight"
        b = normalized.get(b_key)
        module = modules.get(target)
        if b is None or module is None or not hasattr(module, "weight"):
            raise ValueError(f"Unpaired or unmatched LoRA target: {target}")
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != tuple(module.weight.shape):
            raise ValueError(f"Incompatible LoRA shapes: {target}")
        used.update((key, b_key))
    if used != set(normalized) or not used:
        raise ValueError("Unpaired or unsupported LoRA tensors")
    # DiffSynth hot-loads these tensors into plain Python lists owned by its
    # wrapped linear layers. Those lists do not follow ``module.to(...)``, so
    # move the LoRA tensors to the computation device before attaching them.
    normalized = {
        key: value.to(device=pipe.device, dtype=pipe.torch_dtype)
        for key, value in normalized.items()
    }
    adapter_state = {
        key.removeprefix("control_adapter."): value for key, value in adapter.items()
    }
    # VRAM wrappers expose ``*.module.*`` from state_dict(), but their loader
    # intentionally consumes the original unwrapped keys. Exact names/shapes
    # were checked above, so strict=False here only accommodates that wrapper.
    wrapped = any(_logical_module_key(key) != key for key in raw_expected)
    dit.control_adapter.load_state_dict(adapter_state, strict=not wrapped)
    pipe.load_lora(dit, state_dict=normalized, alpha=1.0)


def build_inference_pipeline(config: dict[str, Any], config_path: str | Path) -> Any:
    """Build and checkpoint a pipeline once so batch inference can reuse it."""
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    inference = config["inference"]
    models = config["models"]
    levels = model_levels(config)
    os.environ["HAND_MESH_FACES_PATH"] = str(resolve_path(config_path, config["hand_mesh_faces_path"]))
    device = inference.get("device", "cuda")
    low_vram = bool(inference.get("low_vram", False))
    if low_vram:
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
        }
        total_vram = torch.cuda.mem_get_info(device)[1] / (1024**3)
        vram_limit = total_vram - float(inference.get("vram_margin_gb", 2.0))
        if vram_limit <= 0:
            raise ValueError("vram_margin_gb must be smaller than total GPU memory")
    else:
        vram_config = {"offload_device": "cpu"}
        vram_limit = None

    def model_config(path: str | Path) -> Any:
        return ModelConfig(path=str(resolve_path(config_path, path)), **vram_config)

    model_paths = [models[level]["adapter"] for level in levels]
    model_paths.extend((models["text_encoder"], models["vae"]))
    if models.get("image_encoder"):
        model_paths.append(models["image_encoder"])
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[model_config(path) for path in model_paths],
        tokenizer_config=ModelConfig(path=str(resolve_path(config_path, models["tokenizer"]))),
        vram_limit=vram_limit,
    )
    if levels == ("high", "low"):
        if pipe.dit is None or pipe.dit2 is None:
            raise ValueError("Both high- and low-noise DiT models must be loaded")
        load_finetuned_dit(pipe, pipe.dit, resolve_path(config_path, inference["high_checkpoint"]), "high")
        load_finetuned_dit(pipe, pipe.dit2, resolve_path(config_path, inference["low_checkpoint"]), "low")
    else:
        if pipe.dit is None:
            raise ValueError("The single DiT model was not loaded")
        load_finetuned_dit(pipe, pipe.dit, resolve_path(config_path, inference["checkpoint"]), "single")
    return pipe


def _save_video_atomic(save_video: Any, frames: list[Any], output_path: Path, *, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    try:
        save_video(frames, str(temporary), fps=fps, quality=5)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def run_inference(
    config: dict[str, Any], config_path: str | Path, *, pipe: Any | None = None
) -> Path:
    import numpy as np
    from PIL import Image
    from diffsynth.utils.data import VideoData, save_video

    inference = config["inference"]
    if pipe is None:
        pipe = build_inference_pipeline(config, config_path)

    video = VideoData(
        str(resolve_path(config_path, inference["video"])),
        height=int(inference.get("height", 480)),
        width=inference.get("width"),
    )
    cam_pose = np.load(resolve_path(config_path, inference["cam_pose"]))
    hand_pose = np.load(resolve_path(config_path, inference["hand_pose"]))
    camera_intrinsics = None
    if inference.get("camera_intrinsics"):
        camera_intrinsics = np.load(resolve_path(config_path, inference["camera_intrinsics"]))
    start_frame = int(inference.get("start_frame", 0))
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    available = min(len(video), len(cam_pose), len(hand_pose)) - start_frame
    num_frames = min(int(inference.get("num_frames", 81)), available)
    while num_frames > 0 and num_frames % 4 != 1:
        num_frames -= 1
    if num_frames <= 0:
        raise ValueError("No valid 4k+1 frame sequence is available")
    stop_frame = start_frame + num_frames
    frames = [video[index] for index in range(start_frame, stop_frame)]
    cam_pose = cam_pose[start_frame:stop_frame]
    hand_pose = hand_pose[start_frame:stop_frame]
    if camera_intrinsics is not None and camera_intrinsics.ndim >= 3:
        if len(camera_intrinsics) < stop_frame:
            raise ValueError("Per-frame camera_intrinsics is shorter than the requested frame window")
        camera_intrinsics = camera_intrinsics[start_frame:stop_frame]
    output = pipe(
        prompt=inference["prompt"],
        negative_prompt=inference.get("negative_prompt", ""),
        input_image=frames[0],
        num_frames=num_frames,
        seed=int(inference.get("seed", 0)),
        tiled=bool(inference.get("tiled", True)),
        cam_pose=cam_pose,
        hand_pose=hand_pose,
        camera_intrinsics=camera_intrinsics,
        hand_pose_render_mode=config["representation"],
        hand_pose_attr_space="world",
        original_height=int(inference["original_height"]),
        original_width=int(inference["original_width"]),
        height=int(inference.get("height", frames[0].size[1])),
        width=inference.get("width", frames[0].size[0]),
        cfg_scale=inference.get("cfg_scale", [1.0, 2.0]),
        switch_DiT_boundary=float(inference.get("switch_dit_boundary", 0.9)),
        num_inference_steps=int(inference.get("num_inference_steps", 50)),
    )
    output_path = resolve_path(config_path, inference["output"])
    fps = int(inference.get("fps", 15))
    _save_video_atomic(save_video, output, output_path, fps=fps)
    if inference.get("comparison_output"):
        comparison = []
        for reference, generated in zip(frames, output):
            reference = reference.convert("RGB")
            if not isinstance(generated, Image.Image):
                generated = Image.fromarray(np.asarray(generated))
            generated = generated.convert("RGB")
            if generated.size != reference.size:
                generated = generated.resize(reference.size)
            frame = Image.new("RGB", (reference.width * 2, reference.height))
            frame.paste(reference, (0, 0))
            frame.paste(generated, (reference.width, 0))
            comparison.append(frame)
        comparison_path = resolve_path(config_path, inference["comparison_output"])
        _save_video_atomic(save_video, comparison, comparison_path, fps=fps)
    return output_path


def validate_resources(config: dict[str, Any], config_path: str | Path, purpose: str) -> None:
    """Check local assets before importing the heavy model runtime."""
    models = config["models"]
    levels = model_levels(config)
    files = [config["hand_mesh_faces_path"], models["text_encoder"], models["vae"]]
    if models.get("image_encoder"):
        files.append(models["image_encoder"])
    if purpose == "infer":
        inference = config["inference"]
        files += [models[level]["adapter"] for level in levels]
        checkpoint_keys = ("high_checkpoint", "low_checkpoint") if len(levels) == 2 else ("checkpoint",)
        files += [inference[key] for key in (*checkpoint_keys, "video", "cam_pose", "hand_pose")]
        if inference.get("camera_intrinsics"):
            files.append(inference["camera_intrinsics"])
    else:
        files.append(config["data"]["metadata"])
    missing = [str(resolve_path(config_path, path)) for path in files if not resolve_path(config_path, path).is_file()]
    tokenizer = resolve_path(config_path, models["tokenizer"])
    if not tokenizer.is_dir():
        missing.append(str(tokenizer) + " (tokenizer directory)")
    if missing:
        raise FileNotFoundError("Missing resources: " + ", ".join(missing))
