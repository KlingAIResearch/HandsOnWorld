"""Streaming bridge from official Wan and DiffSynth control checkpoints."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import torch
from safetensors import safe_open
from torch import nn

from diffsynth.models.wan_video_camera_controller import SimpleAdapter

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
except ImportError:  # pragma: no cover - old torch only
    DTensor = ()
    compute_local_shape_and_global_offset = None


class DiffSynthControlCheckpoint:
    """Strict reader for a DiffSynth LoRA plus ``SimpleAdapter`` checkpoint."""

    _ADAPTER_PREFIX = "dit.control_adapter."
    _LORA_A_SUFFIX = ".lora_A.default.weight"

    def __init__(
        self,
        path: str | Path,
        lora_scale: float = 1.0,
        allow_adapter_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.lora_scale = float(lora_scale)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with self.open() as handle:
            keys = set(handle.keys())
            self.adapter_keys = tuple(sorted(k for k in keys if k.startswith(self._ADAPTER_PREFIX)))
            self.lora_keys: dict[str, tuple[str, str]] = {}
            consumed = set(self.adapter_keys)
            for key in sorted(keys):
                if not (key.startswith("dit.") and key.endswith(self._LORA_A_SUFFIX)):
                    continue
                target = key[len("dit."):-len(self._LORA_A_SUFFIX)] + ".weight"
                b_key = key[:-len(self._LORA_A_SUFFIX)] + ".lora_B.default.weight"
                if b_key not in keys:
                    raise ValueError(f"missing LoRA B tensor for {key}")
                a_shape = handle.get_slice(key).get_shape()
                b_shape = handle.get_slice(b_key).get_shape()
                if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                    raise ValueError(f"invalid LoRA pair: {key}, {b_key}")
                self.lora_keys[target] = (key, b_key)
                consumed.update((key, b_key))
            unsupported = sorted(keys - consumed)
            if unsupported:
                raise ValueError(f"unsupported checkpoint keys: {unsupported[:8]}")
            if not self.adapter_keys:
                raise ValueError("checkpoint must contain control-adapter tensors")
            if not self.lora_keys and not allow_adapter_only:
                raise ValueError("checkpoint must contain both LoRA and control-adapter tensors")

    def open(self):
        return safe_open(self.path, framework="pt", device="cpu")

    def validate_base_keys(self, base_keys: Mapping | Sequence[str]) -> None:
        normalized = {_canonical_source_key(key) for key in base_keys}
        missing = sorted(set(self.lora_keys) - normalized)
        if missing:
            raise ValueError(f"LoRA targets are absent from the Wan checkpoint: {missing[:8]}")

    def merge_local(self, handle, source_key, base_slice, slices):
        pair = self.lora_keys.get(_canonical_source_key(source_key))
        if pair is None:
            return base_slice
        if base_slice.ndim != 2 or len(slices) != 2:
            raise ValueError(f"LoRA target must be a 2D weight: {source_key}")
        lora_a = handle.get_slice(pair[0])[:, slices[1]].float()
        lora_b = handle.get_slice(pair[1])[slices[0], :].float()
        delta = lora_b @ lora_a
        if delta.shape != base_slice.shape:
            raise ValueError(f"LoRA delta shape mismatch for {source_key}")
        return torch.add(base_slice.float(), delta, alpha=self.lora_scale)

    def load_adapter(
        self,
        device: torch.device | str | int = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        **adapter_kwargs,
    ) -> SimpleAdapter:
        """Materialize, strictly load and freeze the canonical DiffSynth adapter."""
        aliases = {
            "in_channels": "in_dim",
            "out_channels": "out_dim",
        }
        adapter_kwargs = dict(adapter_kwargs)
        for source, target in aliases.items():
            if source in adapter_kwargs:
                adapter_kwargs[target] = adapter_kwargs.pop(source)
        with torch.device("meta"):
            adapter = SimpleAdapter(**adapter_kwargs)
        adapter.to(dtype=dtype).to_empty(device=device)
        state = adapter.state_dict()
        expected = {self._ADAPTER_PREFIX + key for key in state}
        if expected != set(self.adapter_keys):
            raise ValueError(
                "adapter architecture/checkpoint mismatch; "
                f"missing={sorted(expected-set(self.adapter_keys))[:8]}, "
                f"unexpected={sorted(set(self.adapter_keys)-expected)[:8]}"
            )
        with self.open() as handle, torch.no_grad():
            for key, destination in state.items():
                source = handle.get_tensor(self._ADAPTER_PREFIX + key)
                if source.shape != destination.shape:
                    raise ValueError(f"adapter shape mismatch for {key}")
                destination.copy_(source.to(destination))
        return adapter.eval().requires_grad_(False)


@torch.no_grad()
def encode_control(adapter, packed_control, valid_mask, frames_per_call=1):
    if packed_control.ndim != 5:
        raise ValueError("packed_control must have shape [B,C,T,H,W]")
    if frames_per_call < 1 or packed_control.shape[2] < 1:
        raise ValueError("frames_per_call and frame count must be positive")
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        raise ValueError("the control adapter must be frozen")
    parameter = next(adapter.parameters())
    chunks = []
    for start in range(0, packed_control.shape[2], frames_per_call):
        chunks.append(adapter(packed_control[:, :, start:start + frames_per_call].to(parameter)))
    output = torch.cat(chunks, dim=2)
    mask = valid_mask.to(output)
    if mask.numel() == output.shape[0]:
        mask = mask.reshape(output.shape[0], 1, 1, 1, 1)
    if torch.broadcast_shapes(mask.shape, output.shape) != output.shape:
        raise ValueError("valid_mask does not broadcast to adapter output")
    return output * mask


class OneEntryControlFeatureCache:
    def __init__(self):
        self._key: Hashable | None = None
        self._value: torch.Tensor | None = None

    def get_or_encode(self, key, adapter, packed_control, valid_mask, frames_per_call=1):
        if self._key != key:
            self._value = encode_control(adapter, packed_control, valid_mask, frames_per_call)
            self._key = key
        return self._value

    def clear(self):
        self._key = self._value = None


def _canonical_source_key(key: str) -> str:
    for prefix in ("model.", "module.", "transformer.", "pipe.dit.", "dit."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _weight_map(weights_root: Path) -> dict[str, str]:
    index = weights_root / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]
    shards = sorted(weights_root.glob("diffusion_pytorch_model*.safetensors"))
    if not shards:
        shards = sorted(weights_root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No Wan safetensors shards found under {weights_root}")
    result = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise RuntimeError(f"duplicate Wan tensor {key} in {shard.name} and {result[key]}")
                result[key] = shard.name
    return result


def _local_slice_spec(value):
    if DTensor and isinstance(value, DTensor):
        local = value.to_local()
        shape, offset = compute_local_shape_and_global_offset(value.shape, value.device_mesh, value.placements)
        slices = tuple(slice(start, start + length) for start, length in zip(offset, shape))
        return local, slices, value.numel()
    return value, tuple(slice(0, length) for length in value.shape), value.numel()


def _is_truncated_block(key, max_blocks):
    if max_blocks is None or not key.startswith("blocks."):
        return False
    index = key.split(".", 2)[1]
    return index.isdigit() and int(index) >= max_blocks


@torch.no_grad()
def stream_wan_official_local_state(
    transformer: nn.Module,
    weights_root: str | Path,
    checkpoint: DiffSynthControlCheckpoint | None = None,
    max_blocks: int | None = None,
) -> dict[str, int]:
    """Stream official shards directly into a materialized local/FSDP2 state."""
    weights_root = Path(weights_root)
    weight_map = _weight_map(weights_root)
    if checkpoint is not None:
        checkpoint.validate_base_keys(weight_map)
    canonical_to_source = {}
    for source in weight_map:
        target = _canonical_source_key(source)
        if target in canonical_to_source:
            raise RuntimeError(f"Wan key conversion collision for {target}")
        canonical_to_source[target] = source
    target_state = transformer.state_dict(keep_vars=True)
    missing = sorted(set(target_state) - set(canonical_to_source))
    # logvar is a distillation-only head and is intentionally initialized locally.
    missing_required = [key for key in missing if not key.startswith("logvar_linear.")]
    if missing_required:
        raise RuntimeError(f"model tensors absent from Wan checkpoint: {missing_required[:8]}")
    unexplained = sorted(
        key for key in canonical_to_source
        if key not in target_state and not _is_truncated_block(key, max_blocks)
    )
    if unexplained:
        raise RuntimeError(f"Wan tensors absent from model: {unexplained[:8]}")
    by_file = defaultdict(list)
    for target in target_state:
        if target in canonical_to_source:
            source = canonical_to_source[target]
            by_file[weight_map[source]].append((target, source))
    counters = dict(loaded_tensors=0, loaded_global_numel=0, loaded_local_numel=0, merged_lora_tensors=0)
    checkpoint_context = checkpoint.open() if checkpoint is not None else nullcontext(None)
    with checkpoint_context as checkpoint_handle:
        for filename in sorted(by_file):
            with safe_open(weights_root / filename, framework="pt", device="cpu") as source_handle:
                for target_key, source_key in by_file[filename]:
                    target = target_state[target_key]
                    local, slices, global_numel = _local_slice_spec(target)
                    source_slice = source_handle.get_slice(source_key)
                    if tuple(source_slice.get_shape()) != tuple(target.shape):
                        raise RuntimeError(f"shape mismatch for {target_key}")
                    incoming = source_slice[slices]
                    if checkpoint is not None:
                        incoming = checkpoint.merge_local(checkpoint_handle, source_key, incoming, slices)
                        counters["merged_lora_tensors"] += int(target_key in checkpoint.lora_keys)
                    local.copy_(incoming.to(device=local.device, dtype=local.dtype))
                    counters["loaded_tensors"] += 1
                    counters["loaded_global_numel"] += global_numel
                    counters["loaded_local_numel"] += local.numel()
    return counters


__all__ = [
    "DiffSynthControlCheckpoint", "OneEntryControlFeatureCache", "SimpleAdapter",
    "encode_control", "stream_wan_official_local_state",
]
