"""Prepare portable latent/text/raw-pose caches from public metadata."""

from __future__ import annotations

import json
import csv
import hashlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
import torch

from handsonworld.data.cache import (
    CACHE_FORMAT, CacheDataset, atomic_json, model_cache_spec, sha256_file,
)
from handsonworld.training.config import resolve_path


class SampleEncoder(Protocol):
    def encode(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]: ...


def _distill(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("distillation")
    if not isinstance(value, dict):
        raise ValueError("config.distillation must be an object")
    return value


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except Exception as error:
        raise ValueError(f"Could not convert {name} to a tensor") from error
    return tensor.detach().cpu().contiguous()


def _default_camera_intrinsics(original_height: int, original_width: int) -> torch.Tensor:
    """Match DiffSynth's original-space fallback for missing camera calibration."""
    height, width = int(original_height), int(original_width)
    if height == 1080:
        matrix = torch.tensor([
            [1376.7512207, 0.0, 967.68585205],
            [0.0, 1376.24035645, 526.81701660],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        source_width = 1920
        if width / height < source_width / 1080:
            matrix[0, 2] -= (source_width - width) / 2.0
        return matrix
    return torch.tensor([
        [600.0, 0.0, width / 2.0],
        [0.0, 600.0, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _store_text_tensor(root: Path, value: torch.Tensor) -> str:
    from safetensors.torch import save_file

    value = value.detach().cpu().contiguous()
    digest = _tensor_digest(value)
    relative = Path("texts") / f"{digest}.safetensors"
    destination = root / relative
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.stem + ".tmp.safetensors")
        save_file({"embedding": value}, str(temporary))
        os.replace(temporary, destination)
    return str(relative)


def _source_reference(sample: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in ("video", "cam_pose", "hand_pose", "camera_intrinsics", "prompt", "source"):
        value = sample.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _metadata_references(path: Path, base: Path) -> list[dict[str, Any]]:
    """Read identities before UnifiedDataset replaces paths with decoded arrays."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = json.load(stream) if path.suffix.lower() == ".json" else list(csv.DictReader(stream))
    references = []
    for index, row in enumerate(rows):
        value = row.get("video")
        video = Path(value).expanduser() if value else None
        if video is not None and not video.is_absolute():
            video = (base / video).resolve()
        sample_id = str(index)
        if video is not None:
            try:
                sample_id = str(video.relative_to(base).with_suffix(""))
            except ValueError:
                sample_id = f"{index:06d}_{video.stem}"
        references.append({
            "video": str(video) if video else None,
            "prompt": row.get("prompt", ""), "source": row.get("source"),
            "sample_id": sample_id,
        })
    return references


def _resource_identity(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    result = {}
    models = config.get("models", {})
    for key in ("text_encoder", "vae", "image_encoder"):
        value = models.get(key)
        if not value:
            continue
        path = resolve_path(config_path, value)
        result[key] = {"path": str(path), "size": path.stat().st_size}
    if models.get("tokenizer"):
        tokenizer = resolve_path(config_path, models["tokenizer"])
        result["tokenizer"] = {"path": str(tokenizer)}
    backend_manifest = resolve_path(config_path, config["backend_root"]) / "backend_manifest.json"
    if backend_manifest.is_file():
        result["backend_manifest_sha256"] = sha256_file(backend_manifest)
    return result


def _default_dataset(config: dict[str, Any], config_path: str | Path):
    backend = resolve_path(config_path, config["backend_root"])
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import LoadNpyFile, ToAbsolutePath

    data, training = config["data"], config["training"]
    base = resolve_path(config_path, data["base_path"])
    metadata = resolve_path(config_path, data["metadata"])
    frames = int(training.get("num_frames", 81))
    height, width = int(training.get("height", 480)), int(training.get("width", 640))
    return UnifiedDataset(
        base_path=str(base), metadata_path=str(metadata), repeat=1,
        data_file_keys=("video", "cam_pose", "hand_pose", "camera_intrinsics"),
        random_drop_keys=(), random_start_frame=False, sample_num_frames=frames,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=str(base), max_pixels=int(data.get("max_pixels", 1920 * 1080)),
            height=height, width=width, height_division_factor=32, width_division_factor=32,
            num_frames=frames, time_division_factor=4, time_division_remainder=1,
            resize_mode="resize",
        ),
        special_operator_map={
            "camera_intrinsics": ToAbsolutePath(str(base)) >> LoadNpyFile(),
        },
    )


class _DiffSynthEncoder:
    def __init__(self, config: dict[str, Any], config_path: str | Path) -> None:
        backend = resolve_path(config_path, config["backend_root"])
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        from diffsynth.core import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        settings = _distill(config).get("prepare", {})
        dtype_name = settings.get("dtype", "bfloat16")
        if dtype_name not in {"bfloat16", "float32"}:
            raise ValueError("distillation.prepare.dtype must be bfloat16 or float32")
        self.dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
        self.device = torch.device(settings.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the configured cache encoder device")
        models = config["models"]
        paths = [resolve_path(config_path, models["text_encoder"]), resolve_path(config_path, models["vae"])]
        self.spec = model_cache_spec(config["model_type"])
        if self.spec["requires_clip"]:
            paths.append(resolve_path(config_path, models["image_encoder"]))
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.dtype, device=self.device,
            model_configs=[ModelConfig(path=str(path)) for path in paths],
            tokenizer_config=ModelConfig(path=str(resolve_path(config_path, models["tokenizer"]))),
            audio_processor_config=None, redirect_common_files=False,
        )
        self.pipe.dit = SimpleNamespace(
            require_clip_embedding=self.spec["requires_clip"], require_vae_embedding=self.spec["requires_clip"],
            fuse_vae_embedding_in_latents=not self.spec["requires_clip"], has_image_pos_emb=False,
        )
        self.pipe.eval().requires_grad_(False)
        from diffsynth.pipelines.wan_video import (
            WanVideoUnit_ImageEmbedderCLIP, WanVideoUnit_ImageEmbedderVAE,
            WanVideoUnit_PromptEmbedder,
        )
        self.prompt = WanVideoUnit_PromptEmbedder()
        self.clip = WanVideoUnit_ImageEmbedderCLIP()
        self.first = WanVideoUnit_ImageEmbedderVAE()
        self.negative_prompt = settings.get("negative_prompt", "")
        self.height = int(config["training"].get("height", 480))
        self.width = int(config["training"].get("width", 640))
        self.frames = int(config["training"].get("num_frames", 81))
        self.tiled = bool(settings.get("vae_tiled", False))
        self.tile_size = tuple(settings.get("tile_size", (30, 40)))
        self.tile_stride = tuple(settings.get("tile_stride", (15, 20)))
        self.text_cache_root: Path | None = None

    def set_text_cache_root(self, root: str | Path) -> None:
        self.text_cache_root = Path(root)

    def _encode_text(self, text: str, positive: bool) -> torch.Tensor:
        lookup = None
        if self.text_cache_root is not None:
            identity = json.dumps({
                "text": text, "positive": positive, "dtype": str(self.dtype),
            }, sort_keys=True).encode()
            lookup = self.text_cache_root / "texts" / (
                ".prompt-" + hashlib.sha256(identity).hexdigest() + ".json"
            )
            try:
                record = json.loads(lookup.read_text(encoding="utf-8"))
                from safetensors.torch import load_file
                cached = load_file(str(self.text_cache_root / record["file"]), device="cpu")["embedding"]
                return cached.to(device=self.device, dtype=self.dtype)
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        value = self.prompt.process(self.pipe, prompt=text, positive=positive)["context"][0]
        if lookup is not None:
            relative = _store_text_tensor(self.text_cache_root, value)
            atomic_json(lookup, {"file": relative})
        return value

    @torch.inference_mode()
    def encode(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        video = sample["video"]
        if len(video) != self.frames:
            raise ValueError(f"Decoded video has {len(video)} frames, expected {self.frames}")
        video_tensor = self.pipe.preprocess_video(video)
        real = self.pipe.vae.encode(
            video_tensor, device=self.device, tiled=self.tiled,
            tile_size=self.tile_size, tile_stride=self.tile_stride,
        ).to(device=self.device, dtype=self.dtype)
        prompt = sample.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = ""
        condition = self._encode_text(prompt, True)
        negative = self._encode_text(self.negative_prompt, False)
        result = {
            "real": real[0], "condition": condition, "neg_condition": negative,
        }
        if self.spec["requires_clip"]:
            result["encoder_hidden_states_image"] = self.clip.process(
                self.pipe, input_image=video[0], end_image=None, height=self.height, width=self.width,
            )["clip_feature"][0]
            y = self.first.process(
                self.pipe, input_image=video[0], end_image=None, num_frames=self.frames,
                height=self.height, width=self.width, tiled=self.tiled,
                tile_size=self.tile_size, tile_stride=self.tile_stride,
            )["y"]
            if y.shape[1] != 20:
                raise ValueError(f"Wan2.1 first-frame embedding must have 20 channels, got {y.shape[1]}")
            result["first_frame_cond"] = y[0, 4:]
        else:
            image = self.pipe.preprocess_image(video[0].resize((self.width, self.height))).transpose(0, 1)
            first = self.pipe.vae.encode(
                [image], device=self.device, tiled=self.tiled,
                tile_size=self.tile_size, tile_stride=self.tile_stride,
            )[0].to(device=self.device, dtype=self.dtype)
            padded = torch.zeros_like(real[0])
            padded[:, :1] = first[:, :1]
            result["first_frame_cond"] = padded
        return {key: value.detach().cpu().contiguous() for key, value in result.items()}


def prepare(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    dataset=None,
    encoder: SampleEncoder | None = None,
) -> Path:
    """Encode source samples and atomically publish a complete cache manifest.

    ``dataset`` and ``encoder`` are injectable to make the data contract
    testable without loading GPU models.
    """
    distill = _distill(config)
    root = Path(distill["cache_dir"]).expanduser().resolve()
    # Cache encoding is deliberately a single-writer stage.  torchrun peers
    # may exit normally while rank zero completes; the parent launcher still
    # waits for rank zero and therefore publishes no early success.
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return root / "index.json"
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    data, training = config["data"], config["training"]
    metadata = resolve_path(config_path, data["metadata"])
    video_shape = [
        int(training.get("num_frames", 81)), int(training.get("height", 480)),
        int(training.get("width", 640)),
    ]
    prepare_settings = {**distill, **distill.get("prepare", {})}
    tile_size = list(prepare_settings.get("tile_size", (30, 40)))
    tile_stride = list(prepare_settings.get("tile_stride", (15, 20)))
    provenance = {
        "metadata": str(metadata), "metadata_sha256": sha256_file(metadata),
        "model_type": config["model_type"], "video_shape": video_shape,
        "sample_window": {"start_frame": 0, "random_start": False},
        "resize": {"mode": "resize", "height": video_shape[1], "width": video_shape[2]},
        "intrinsics_space": "source_with_original_size",
        "encoding": {
            "negative_prompt": prepare_settings.get("negative_prompt", ""),
            "dtype": prepare_settings.get("dtype", "bfloat16"),
            "vae_tiled": bool(prepare_settings.get("vae_tiled", False)),
            "tile_size": tile_size, "tile_stride": tile_stride,
            "seed": int(prepare_settings.get("seed", 0)),
        },
        "selection": {
            "start_index": int(prepare_settings.get("start_index", 0)),
            "max_samples": int(prepare_settings.get("max_samples", 0)),
        },
        "resources": _resource_identity(config, config_path),
    }
    previous = None
    if index_path.is_file():
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        if previous.get("provenance") != provenance:
            raise ValueError("Existing cache provenance differs; use a new cache_dir")
    if dataset is None:
        dataset = _default_dataset(config, config_path)
    if encoder is None:
        encoder = _DiffSynthEncoder(config, config_path)
    if hasattr(encoder, "set_text_cache_root"):
        encoder.set_text_cache_root(root)
    settings = prepare_settings
    start = int(settings.get("start_index", 0))
    maximum = int(settings.get("max_samples", 0))
    stop = len(dataset) if maximum <= 0 else min(len(dataset), start + maximum)
    if not 0 <= start < stop:
        raise ValueError(f"Empty cache export range [{start}, {stop})")
    seed = int(settings.get("seed", 0))
    references = _metadata_references(metadata, resolve_path(config_path, data["base_path"]))
    manifest = {
        "format": CACHE_FORMAT, "schema_version": 1, "complete": False,
        "model_type": config["model_type"], "video_shape": video_shape,
        "control_representation": "raw",
        "compatible_control_representations": ["normal_plucker", "reflection_plucker"],
        "provenance": provenance,
        "samples": list(previous.get("samples", [])) if previous else [],
    }
    atomic_json(index_path, manifest)
    from safetensors.torch import save_file

    known = {int(entry["source_index"]): entry for entry in manifest["samples"]}
    for source_index in range(start, stop):
        name = f"sample-{source_index:08d}.safetensors"
        destination = root / name
        if source_index in known and destination.is_file():
            continue
        _seed(seed + source_index)
        sample = dataset[source_index]
        if getattr(sample["video"], "original_size", None) is not None:
            from diffsynth.core.data.operators import resolve_original_video_size

            original_height, original_width = resolve_original_video_size(
                sample["video"], sample.get("height"), sample.get("width"),
            )
        else:
            original_height = int(sample.get("height") or video_shape[1])
            original_width = int(sample.get("width") or video_shape[2])
        tensors = encoder.encode(sample)
        text_references = {
            key: _store_text_tensor(root, tensors.pop(key))
            for key in ("condition", "neg_condition")
        }
        camera_intrinsics = sample.get("camera_intrinsics")
        if camera_intrinsics is None:
            camera_intrinsics = _default_camera_intrinsics(original_height, original_width)
        tensors.update(
            raw_cam_pose=_as_tensor(sample["cam_pose"], "cam_pose").float(),
            raw_hand_pose=_as_tensor(sample["hand_pose"], "hand_pose").float(),
            raw_camera_intrinsics=_as_tensor(camera_intrinsics, "camera_intrinsics").float(),
            raw_original_height=torch.tensor(original_height, dtype=torch.int64),
            raw_original_width=torch.tensor(original_width, dtype=torch.int64),
        )
        temporary = destination.with_name(destination.stem + ".tmp.safetensors")
        save_file(tensors, str(temporary), metadata={"source_index": str(source_index)})
        os.replace(temporary, destination)
        reference = references[source_index] if source_index < len(references) else _source_reference(sample)
        entry = {
            "file": name, "source_index": source_index,
            "sample_id": str(reference.get("sample_id", sample.get("sample_id", source_index))),
            "source": reference, "seed": seed + source_index, "text": text_references,
        }
        known[source_index] = entry
        manifest["samples"] = [known[key] for key in sorted(known)]
        atomic_json(index_path, manifest)
    manifest["complete"] = True
    atomic_json(index_path, manifest)
    # Fresh validation is the publication gate.
    CacheDataset(root, config["model_type"])
    return index_path
