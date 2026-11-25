"""Validated, portable conditioning cache for causal Wan training.

The cache intentionally stores raw camera/hand inputs rather than frozen
adapter features.  This keeps normal/reflection recipes interchangeable and
lets training use the one shared control adapter implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from handsonworld.training.config import ConfigError, MODEL_SPECS


CACHE_FORMAT = "handsonworld_condition_cache_v1"
SUPPORTED_MODELS = ("wan2.1-i2v-14b-480p", "wan2.2-ti2v-5b")
COMMON_TENSORS = (
    "real", "condition", "neg_condition", "first_frame_cond",
    "raw_cam_pose", "raw_hand_pose", "raw_camera_intrinsics",
    "raw_original_height", "raw_original_width",
)


def model_cache_spec(model_type: str) -> dict[str, Any]:
    if model_type not in SUPPORTED_MODELS:
        raise ConfigError(
            f"Self-Forcing cache model_type must be one of {SUPPORTED_MODELS}, got {model_type!r}"
        )
    base = MODEL_SPECS[model_type]
    is_21 = model_type == "wan2.1-i2v-14b-480p"
    return {
        "model_type": model_type,
        "latent_channels": 16 if is_21 else 48,
        "spatial_downscale": int(base["downscale_factor"]),
        "requires_clip": is_21,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_sample_path(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"Unsafe cache sample filename: {name!r}")
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Cache sample escapes its root: {name!r}") from error
    if path.suffix != ".safetensors":
        raise ValueError(f"Cache sample must be .safetensors: {name!r}")
    return path


def _safe_text_path(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError(f"Unsafe text cache filename: {name!r}")
    path = (root / name).resolve()
    text_root = (root / "texts").resolve()
    try:
        path.relative_to(text_root)
    except ValueError as error:
        raise ValueError(f"Text cache file escapes texts/: {name!r}") from error
    if path.suffix != ".safetensors" or not path.is_file():
        raise ValueError(f"Missing text cache tensor: {path}")
    return path


class CacheDataset(Dataset):
    """Read a completed public cache and validate its tensor/model contract."""

    def __init__(
        self,
        path: str | Path,
        model_type: str,
        *,
        expected_control_representation: str | None = None,
        require_complete: bool = True,
        include_metadata: bool = False,
    ) -> None:
        self.root = Path(path).expanduser().resolve()
        self.include_metadata = include_metadata
        self.index_path = self.root / "index.json"
        try:
            self.manifest = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Failed to read cache manifest {self.index_path}: {error}") from error
        if self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError(f"Cache format must be {CACHE_FORMAT!r}")
        if self.manifest.get("model_type") != model_type:
            raise ValueError(
                f"Cache model_type {self.manifest.get('model_type')!r} does not match {model_type!r}"
            )
        self.spec = model_cache_spec(model_type)
        self.manifest_sha256 = sha256_file(self.index_path)
        representation = self.manifest.get("control_representation", "raw")
        compatible = self.manifest.get(
            "compatible_control_representations", ["normal_plucker", "reflection_plucker"]
        )
        if representation != "raw" or set(compatible) != {"normal_plucker", "reflection_plucker"}:
            raise ValueError("Public caches must contain representation-neutral raw controls")
        if expected_control_representation and expected_control_representation not in compatible:
            raise ValueError(
                f"Cache does not support control representation {expected_control_representation!r}"
            )
        if require_complete and self.manifest.get("complete") is not True:
            raise ValueError(f"Cache is incomplete: {self.index_path}")
        video_shape = self.manifest.get("video_shape")
        if (
            not isinstance(video_shape, list) or len(video_shape) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in video_shape)
        ):
            raise ValueError("Cache video_shape must contain three positive integers")
        frames, height, width = video_shape
        if frames % 4 != 1:
            raise ValueError("Cache frame count must be 1 mod 4")
        scale = self.spec["spatial_downscale"]
        if height % scale or width % scale:
            raise ValueError(f"Cache resolution must be divisible by {scale}")
        self.video_shape = tuple(video_shape)
        self.latent_shape = (
            self.spec["latent_channels"], (frames - 1) // 4 + 1, height // scale, width // scale,
        )
        samples = self.manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Cache manifest must contain samples")
        self.samples: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for position, entry in enumerate(samples):
            if not isinstance(entry, dict):
                raise ValueError(f"Cache sample {position} must be an object")
            sample_path = _safe_sample_path(self.root, entry.get("file"))
            if sample_path in seen or not sample_path.is_file():
                raise ValueError(f"Missing or duplicate cache sample: {sample_path}")
            seen.add(sample_path)
            text = entry.get("text")
            text_paths = None
            if text is not None:
                if not isinstance(text, dict) or set(text) != {"condition", "neg_condition"}:
                    raise ValueError(f"Cache sample {position} has invalid text references")
                text_paths = {key: _safe_text_path(self.root, value) for key, value in text.items()}
            self.samples.append({**entry, "path": sample_path, "text_paths": text_paths})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from safetensors.torch import load_file

        entry = self.samples[index]
        tensors = load_file(str(entry["path"]), device="cpu")
        if entry["text_paths"] is not None:
            if "condition" in tensors or "neg_condition" in tensors:
                raise ValueError(f"Cache sample mixes inline and referenced text: {entry['path']}")
            for key, path in entry["text_paths"].items():
                values = load_file(str(path), device="cpu")
                if set(values) != {"embedding"}:
                    raise ValueError(f"Text cache tensor must contain only embedding: {path}")
                tensors[key] = values["embedding"]
        self._validate_tensors(tensors, entry["path"])
        result: dict[str, Any] = dict(tensors)
        if self.include_metadata:
            result.update(
                sample_index=int(entry.get("source_index", index)),
                sample_id=str(entry.get("sample_id", entry["path"].stem)),
                source_reference=entry.get("source"),
            )
        return result

    def _validate_tensors(self, tensors: Mapping[str, torch.Tensor], path: Path) -> None:
        required = set(COMMON_TENSORS)
        if self.spec["requires_clip"]:
            required.add("encoder_hidden_states_image")
        missing = required - set(tensors)
        if missing:
            raise ValueError(f"Cache sample {path} is missing tensors: {sorted(missing)}")
        if not self.spec["requires_clip"] and "encoder_hidden_states_image" in tensors:
            raise ValueError(f"Wan2.2 TI2V 5B cache must not contain CLIP features: {path}")
        for key in ("real", "first_frame_cond"):
            value = tensors[key]
            if tuple(value.shape) != self.latent_shape:
                raise ValueError(f"{key} shape {tuple(value.shape)} != {self.latent_shape} in {path}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{key} must be finite floating point in {path}")
        if not self.spec["requires_clip"] and torch.count_nonzero(tensors["first_frame_cond"][:, 1:]):
            raise ValueError(f"Wan2.2 first_frame_cond must be zero after its first latent frame in {path}")
        for key in ("condition", "neg_condition"):
            value = tensors[key]
            if value.ndim != 2 or value.shape[-1] != 4096 or not torch.isfinite(value).all():
                raise ValueError(f"{key} must have shape [tokens,4096] and be finite in {path}")
        if self.spec["requires_clip"]:
            clip = tensors["encoder_hidden_states_image"]
            if clip.ndim != 2 or clip.shape[-1] != 1280 or not torch.isfinite(clip).all():
                raise ValueError(f"encoder_hidden_states_image must be [tokens,1280] in {path}")
        frames = self.video_shape[0]
        cam, hand, intrinsics = (
            tensors["raw_cam_pose"], tensors["raw_hand_pose"], tensors["raw_camera_intrinsics"]
        )
        if tuple(cam.shape) != (frames, 4, 4):
            raise ValueError(f"raw_cam_pose must be [{frames},4,4] in {path}")
        if hand.ndim != 3 or hand.shape[0] != frames or hand.shape[-1] != 3:
            raise ValueError(f"raw_hand_pose must be [{frames},points,3] in {path}")
        if tuple(intrinsics.shape) != (3, 3):
            raise ValueError(f"raw_camera_intrinsics must be [3,3] in {path}")
        for dimension in ("raw_original_height", "raw_original_width"):
            if dimension in tensors and (tensors[dimension].numel() != 1 or tensors[dimension].item() <= 0):
                raise ValueError(f"{dimension} must be a positive scalar in {path}")
        for key in ("raw_cam_pose", "raw_hand_pose", "raw_camera_intrinsics"):
            if not torch.isfinite(tensors[key]).all():
                raise ValueError(f"{key} contains non-finite values in {path}")


def validate_timesteps(values: Sequence[float] | torch.Tensor, name: str = "timesteps") -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    parsed = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} contains a non-finite value")
    if any(left <= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError(f"{name} must be strictly decreasing")
    return parsed
