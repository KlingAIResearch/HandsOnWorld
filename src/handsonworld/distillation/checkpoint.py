"""Sharded, sequential DCP checkpoints; no full-model state dict on rank zero."""

from pathlib import Path
import json
import os
import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


class ModelWrapper(Stateful):
    """Expose a model through DCP's stateful protocol."""

    def __init__(self, model, options: StateDictOptions | None = None):
        self.model = model
        self.options = options

    def state_dict(self):
        return get_model_state_dict(self.model, options=self.options)

    def load_state_dict(self, state_dict):
        set_model_state_dict(self.model, model_state_dict=state_dict, options=self.options)


class OptimizerWrapper(Stateful):
    """Expose an optimizer and its FSDP model through DCP."""

    def __init__(self, model, optimizer, options: StateDictOptions | None = None):
        self.model = model
        self.optimizer = optimizer
        self.options = options

    def state_dict(self):
        return get_optimizer_state_dict(self.model, self.optimizer, options=self.options)

    def load_state_dict(self, state_dict):
        set_optimizer_state_dict(
            self.model, self.optimizer, optim_state_dict=state_dict, options=self.options
        )


def contracts_match_for_resume(saved, current):
    """Allow only relocation of a manifest-identified text cache."""
    if saved == current:
        return True
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    saved_digest = saved.get("text_cache_manifest_sha256")
    current_digest = current.get("text_cache_manifest_sha256")
    if not saved_digest or saved_digest != current_digest:
        return False
    saved_copy, current_copy = dict(saved), dict(current)
    saved_copy.pop("text_cache", None)
    current_copy.pop("text_cache", None)
    return saved_copy == current_copy


def plain_state(value):
    """Scheduler state may contain NumPy scalars; keep weights_only loads safe."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(plain_state(item) for item in value)
    return value


def load_student(network, directory):
    directory = Path(directory)
    if not (directory / "complete.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {directory}")
    dcp.load({"model": ModelWrapper(network)}, checkpoint_id=directory / "net")


def load_unsharded_student(network, directory):
    """Fill ordinary parameter storage without retaining the model in DCP's FQN cache."""
    directory = Path(directory)
    if not (directory / "complete.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {directory}")
    dcp.load({"model": network.state_dict()}, checkpoint_id=directory / "net")


@torch.no_grad()
def load_safetensors_student(network, filename):
    """Stream a portable student bundle into already-materialized storage."""
    filename = Path(filename)
    # state_dict(keep_vars=True) selects the persistent checkpoint contract
    # while retaining direct references to destination storage. In particular,
    # non-persistent scheduler/RoPE buffers must not be required from bundles.
    destinations = network.state_dict(keep_vars=True)
    with safe_open(filename, framework="pt", device="cpu") as handle:
        source_keys = set(handle.keys())
        expected_keys = set(destinations)
        if source_keys != expected_keys:
            raise ValueError(
                "portable student/model mismatch; "
                f"missing={sorted(expected_keys-source_keys)[:8]}, "
                f"unexpected={sorted(source_keys-expected_keys)[:8]}"
            )
        for key, destination in destinations.items():
            source = handle.get_tensor(key)
            if source.shape != destination.shape:
                raise ValueError(
                    f"portable student shape mismatch for {key}: "
                    f"{tuple(source.shape)} != {tuple(destination.shape)}"
                )
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def save_checkpoint(directory, model, optimizers, schedulers, iteration, contract, *, preallocated=False,
                    save_optimizer=True):
    from handsonworld.distillation.checkpoint_reservation import (
        ReservedFileSystem, activate_reservation, collective_call, reserved_writer,
    )

    directory = Path(directory)
    def prepare_directory():
        if dist.get_rank() == 0:
            if directory.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint {directory}")
            if preallocated:
                activate_reservation(directory, contract)
            else:
                directory.mkdir(parents=True)
    collective_call(prepare_directory)
    # Serialize one role at a time, without gathering or CPU-offloading the
    # entire state dict. FileSystemWriter stages individual local tensors.
    for role, optimizer in optimizers.items():
        network = getattr(model, role)
        writer = {"storage_writer": reserved_writer(directory / role)} if preallocated else {}
        dcp.save({"model": ModelWrapper(network)}, checkpoint_id=directory / role, **writer)
        if save_optimizer and optimizer.state:
            path = directory / f"{role}_optimizer"
            writer = {"storage_writer": reserved_writer(path)} if preallocated else {}
            dcp.save({"optimizer": OptimizerWrapper(network, optimizer)}, checkpoint_id=path, **writer)
    rank_state = {
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "schedulers": {role: plain_state(scheduler.state_dict()) for role, scheduler in schedulers.items()},
    } if save_optimizer else None
    def save_rank():
        path = directory / f"rank-{dist.get_rank()}.pt"
        if preallocated:
            with ReservedFileSystem().create_stream(path, "wb") as handle:
                torch.save(rank_state, handle)
        else:
            torch.save(rank_state, path)
    if save_optimizer:
        collective_call(save_rank)

    def publish():
        if dist.get_rank() != 0:
            return
        metadata = {
            "iteration": iteration,
            "world_size": dist.get_world_size(),
            "contract": contract,
            "optimizer_roles": [role for role, opt in optimizers.items() if save_optimizer and opt.state],
        }
        if not save_optimizer:
            metadata["training_state_complete"] = False
            metadata["weights_only"] = True
        temporary = directory / "complete.json.tmp"
        payload = (json.dumps(metadata, indent=2) + "\n").encode()
        if preallocated:
            with ReservedFileSystem().create_stream(temporary, "wb") as handle:
                handle.write(payload)
        else:
            temporary.write_bytes(payload)
        os.replace(temporary, directory / "complete.json")
    collective_call(publish)


def resume_checkpoint(directory, model, optimizers, schedulers, contract):
    directory = Path(directory)
    metadata = json.loads((directory / "complete.json").read_text())
    if metadata.get("training_state_complete") is False:
        raise ValueError("Checkpoint has no optimizer/RNG state for exact resume; use --student-init, not --resume")
    if not contracts_match_for_resume(metadata["contract"], contract):
        raise ValueError(
            "Resume contract differs (stage, data, weights, precision, shape or schedule); use --student-init for a new stage"
        )
    if metadata["world_size"] != dist.get_world_size():
        raise ValueError("Exact resume requires the same world size; --student-init supports resharded stage transfer")
    for role, optimizer in optimizers.items():
        network = getattr(model, role)
        dcp.load({"model": ModelWrapper(network)}, checkpoint_id=directory / role)
        if role in metadata["optimizer_roles"]:
            # DCP initializes empty Adam states through zero-gradient steps.
            # fake_score may otherwise be frozen after the preceding phase.
            network.requires_grad_(True)
            dcp.load({"optimizer": OptimizerWrapper(network, optimizer)}, checkpoint_id=directory / f"{role}_optimizer")
            optimizer.zero_grad(set_to_none=True)
    state = torch.load(directory / f"rank-{dist.get_rank()}.pt", map_location="cpu", weights_only=True)
    for role, scheduler in schedulers.items():
        scheduler.load_state_dict(state["schedulers"][role])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return metadata["iteration"]
