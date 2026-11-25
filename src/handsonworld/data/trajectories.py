"""Validated teacher ODE trajectories joined to a public condition cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .cache import CacheDataset, validate_timesteps


ODE_FORMAT = "handsonworld_ode_trajectory_v1"


class ODEDataset(Dataset):
    def __init__(
        self,
        index_file: str | Path,
        model_type: str,
        *,
        cache_dir: str | Path | None = None,
        expected_path_timesteps: Sequence[float] | None = None,
        expected_control_representation: str | None = None,
    ) -> None:
        self.index_file = Path(index_file).expanduser().resolve()
        self.root = self.index_file.parent
        try:
            self.manifest = json.loads(self.index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Failed to read ODE manifest {self.index_file}: {error}") from error
        if self.manifest.get("format") != ODE_FORMAT or self.manifest.get("complete") is not True:
            raise ValueError(f"ODE manifest is incomplete or has the wrong format: {self.index_file}")
        if self.manifest.get("model_type") != model_type:
            raise ValueError("ODE model_type does not match the requested model")
        representation = self.manifest.get("control_representation")
        target_representation = self.manifest.get("target_control_representation", representation)
        if expected_control_representation and target_representation != expected_control_representation:
            raise ValueError("ODE target control representation does not match the training recipe")
        cross_control = bool(self.manifest.get("allow_cross_control", False))
        if representation != target_representation and not cross_control:
            raise ValueError("Cross-control ODE requires allow_cross_control=true")
        configured_cache = Path(self.manifest.get("cache", ""))
        if not configured_cache.is_absolute():
            configured_cache = (self.root / configured_cache).resolve()
        if cache_dir is not None and Path(cache_dir).expanduser().resolve() != configured_cache:
            raise ValueError("ODE cache reference differs from the configured cache_dir")
        self.cache = CacheDataset(
            configured_cache, model_type,
            expected_control_representation=representation,
        )
        expected_cache_digest = self.manifest.get("cache_manifest_sha256")
        if expected_cache_digest != self.cache.manifest_sha256:
            raise ValueError("ODE cache manifest digest does not match the referenced cache")
        student_digest = self.manifest.get("source_student_checkpoint_digest")
        if (
            not isinstance(student_digest, str) or len(student_digest) != 64
            or any(char not in "0123456789abcdef" for char in student_digest)
        ):
            raise ValueError("ODE source student checkpoint digest must be a lowercase SHA-256")
        actual = validate_timesteps(self.manifest.get("path_timesteps"), "ODE path_timesteps")
        if expected_path_timesteps is not None:
            expected = validate_timesteps(expected_path_timesteps, "expected path_timesteps")
            if len(actual) != len(expected) or any(abs(a - b) > 1e-6 for a, b in zip(actual, expected)):
                raise ValueError(f"ODE path timesteps {actual} do not match expected {expected}")
        self.path_timesteps = actual
        samples = self.manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("ODE manifest must contain samples")
        self.samples = []
        seen: set[int] = set()
        for position, entry in enumerate(samples):
            if not isinstance(entry, dict):
                raise ValueError(f"ODE sample {position} must be an object")
            source_index = entry.get("cache_index")
            if isinstance(source_index, bool) or not isinstance(source_index, int):
                raise ValueError(f"ODE sample {position} cache_index must be an integer")
            if source_index in seen or not 0 <= source_index < len(self.cache):
                raise ValueError(f"ODE sample {position} has duplicate or invalid cache_index")
            seen.add(source_index)
            relative = entry.get("file")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError(f"ODE sample {position} has an unsafe filename")
            path = (self.root / relative).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as error:
                raise ValueError(f"ODE sample escapes output root: {relative}") from error
            if path.suffix != ".safetensors" or not path.is_file():
                raise ValueError(f"Missing ODE trajectory: {path}")
            self.samples.append({**entry, "path": path})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from safetensors.torch import load_file

        entry = self.samples[index]
        generated = load_file(str(entry["path"]), device="cpu")
        self._validate(generated, entry["path"])
        result = self.cache[entry["cache_index"]]
        result.update(generated)
        result["trajectory_seed"] = int(entry.get("seed", 0))
        return result

    def _validate(self, values: Mapping[str, torch.Tensor], path: Path) -> None:
        if set(values) != {"real", "path", "path_timesteps"}:
            raise ValueError(f"ODE trajectory {path} must contain real, path and path_timesteps only")
        latent_shape = self.cache.latent_shape
        if tuple(values["real"].shape) != latent_shape:
            raise ValueError(f"ODE endpoint shape mismatch in {path}")
        if tuple(values["path"].shape) != (len(self.path_timesteps), *latent_shape):
            raise ValueError(f"ODE path shape mismatch in {path}")
        actual = validate_timesteps(values["path_timesteps"], f"ODE timesteps in {path}")
        if len(actual) != len(self.path_timesteps) or any(
            abs(a - b) > 1e-6 for a, b in zip(actual, self.path_timesteps)
        ):
            raise ValueError(f"ODE trajectory timesteps differ from its manifest: {path}")
        for key in ("real", "path"):
            if not values[key].is_floating_point() or not torch.isfinite(values[key]).all():
                raise ValueError(f"ODE {key} must be finite floating point in {path}")


# Backward-friendly public name used by the migrated training code.
WanI2VODEDataset = ODEDataset
