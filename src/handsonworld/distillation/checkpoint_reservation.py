"""Prewrite future DCP files and consume their allocated inodes at save time.

Uses the installed DCP planner for replica deduplication. Planning reads tensor
metadata only; it must never stage model weights or initialize Adam moments.
"""

from collections import defaultdict
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import pickle

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import DefaultSavePlanner, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.planner import WriteItemType

from handsonworld.distillation.checkpoint import ModelWrapper, OptimizerWrapper
from handsonworld.distillation.checkpoint_schedule import checkpoint_steps
from handsonworld.distillation.reserved_output import reserve_output, reservation_path


MIB = 1024 * 1024


def metadata_capacity(metadata, plans):
    # Planning metadata has NO storage_data. FileSystemWriter.finish adds one
    # MetadataIndex -> _StorageInfo record per write item, plus storage_meta.
    # Doubling the planning pickle was insufficient for the full 14B optimizer.
    # Budget these records explicitly, with a floor for serialization overhead.
    entries = sum(len(plan.items) for plan in plans)
    return 64 * MIB + 4 * len(pickle.dumps(metadata)) + 4096 * entries


def save_steps(initial, total, every):
    """Compatibility wrapper for the original interval-only helper."""
    return checkpoint_steps(initial, total, every)


def collective_call(function):
    """Propagate local filesystem/planning failures before the next collective."""
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Checkpoint reservation failed: " + "; ".join(e for e in errors if e))
    return value


def acquire_output_lock(output):
    """Rank zero owns this flock until training exits, including during saves."""
    def acquire():
        if dist.get_rank() != 0:
            return None
        Path(output).mkdir(parents=True, exist_ok=True)
        handle = open(Path(output) / ".training.lock", "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError(f"Another training job owns output directory {output}") from None
        return handle
    return collective_call(acquire)


def optimizer_blueprint(network, optimizer):
    """Borrow parameter shapes for uninitialized Adam slots, without an update.

    Aliases are used exclusively for planning. No moment storage is allocated,
    and the original optimizer state (including its identity) is restored.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError("Checkpoint reservation currently supports Adam/AdamW only")
    original = optimizer.state
    try:
        optimizer.state = defaultdict(dict, original)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if optimizer.state.get(parameter):
                    continue
                alias = parameter.detach()
                state = dict(step=torch.tensor(0.0), exp_avg=alias, exp_avg_sq=alias)
                if group.get("amsgrad", False):
                    state["max_exp_avg_sq"] = alias
                optimizer.state[parameter] = state
        return OptimizerWrapper(network, optimizer).state_dict()
    finally:
        optimizer.state = original


def plan_capacity(state):
    """Return this rank's DCP file bound and the coordinator metadata bound."""
    planner = DefaultSavePlanner()

    def local_plan():
        planner.set_up_planner(state, is_coordinator=dist.get_rank() == 0)
        return planner.create_local_plan()

    local = collective_call(local_plan)
    plans = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local, plans, dst=0)

    def global_plan():
        if dist.get_rank() == 0:
            planned, metadata = planner.create_global_plan(plans)
            return planned, metadata_capacity(metadata, planned)
        return None

    global_result = collective_call(global_plan)
    received = [None]
    dist.scatter_object_list(received, global_result[0] if global_result else None, src=0)
    plan = received[0]

    def size_bound():
        size = MIB
        for item in plan.items:
            if item.type == WriteItemType.BYTE_IO:
                size += planner.resolve_data(item).getbuffer().nbytes
            else:
                tensor = item.tensor_data
                size += math.prod(tensor.chunk.sizes) * torch.empty((), dtype=tensor.properties.dtype).element_size()
            # torch.save ZIP headers/alignment; leave room per independent item.
            size += 16 * 1024
        return size

    capacity = collective_call(size_bound)
    return capacity, global_result[1] if global_result else None


def recover_incomplete(directory, slot, identity):
    """Adopt only our own incomplete save, retaining already allocated inodes."""
    if (directory / "complete.json").exists() or slot.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint/conflicting reservation {directory}")
    manifest = directory / "reservation.json"
    if not manifest.is_file() or json.loads(manifest.read_text()) != identity:
        raise ValueError(f"Incomplete checkpoint has no matching reservation manifest: {directory}")
    allowed = {"reservation.json"}
    moves = []
    for relative, size in identity["files"].items():
        final_names = [relative]
        if relative.endswith("/.metadata.tmp"):
            final_names.append(relative.removesuffix(".tmp"))
        names = [*final_names, relative + ".reserve", relative + ".reserve.allocating"]
        allowed.update(names)
        existing = [name for name in names if (directory / name).exists()]
        if len(existing) > 1:
            raise ValueError(f"Conflicting incomplete checkpoint files for {relative}")
        if not existing:
            continue
        name = existing[0]
        actual_size = (directory / name).stat().st_size
        if actual_size > size:
            raise ValueError(f"Incomplete checkpoint file exceeds reservation: {name}")
        if name.endswith(".reserve.allocating") or (name.endswith(".reserve") and actual_size == size):
            continue
        moves.append((name, relative + ".reserve.allocating"))
    unknown = {str(f.relative_to(directory)) for f in directory.rglob("*") if f.is_file()} - allowed
    if unknown:
        raise ValueError(f"Unexpected files in incomplete checkpoint {directory}: {sorted(unknown)}")
    if any(f.stat().st_nlink > 1 for f in directory.rglob("*") if f.is_file()):
        raise ValueError("Incomplete checkpoint weights have been exported; use a new OUTPUT to preserve them")
    # Validate everything first, then rename in place. No allocated file is deleted.
    for source, target in moves:
        os.rename(directory / source, directory / target)
    slot.parent.mkdir(parents=True, exist_ok=True)
    os.rename(directory, slot)


def prepare_manifest(slot, identity):
    """Allow only metadata-capacity growth; reuse all existing tensor files."""
    manifest = slot / "reservation.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous == identity:
            return
        same_contract = (
            {k: v for k, v in previous.items() if k != "files"}
            == {k: v for k, v in identity.items() if k != "files"}
        )
        old_files, new_files = previous.get("files", {}), identity["files"]
        changed = [name for name in new_files if old_files.get(name) != new_files[name]]
        if not same_contract or old_files.keys() != new_files.keys() or any(
            not name.endswith("/.metadata.tmp") or new_files[name] < old_files[name]
            for name in changed
        ):
            raise ValueError(f"Reservation layout/contract changed: {slot}; use a new OUTPUT")
        for name in changed:
            ready = reservation_path(slot / name)
            allocating = ready.with_name(ready.name + ".allocating")
            if ready.exists():
                if allocating.exists() or ready.stat().st_size != old_files[name]:
                    raise ValueError(f"Conflicting metadata reservation: {ready}")
            elif allocating.exists() and allocating.stat().st_size > new_files[name]:
                raise ValueError(f"Oversized partial metadata reservation: {allocating}")
        for name in changed:
            ready = reservation_path(slot / name)
            if ready.exists():
                os.rename(ready, ready.with_name(ready.name + ".allocating"))
        print(f"CHECKPOINT upgrading metadata reservations in {slot}; tensor reservations reused", flush=True)
    temporary = slot / "reservation.json.tmp"
    with temporary.open("w") as handle:
        json.dump(identity, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest)


def reserve_checkpoints(
    output, model, optimizers, initial, total, every, contract, *, save_optimizer=True, save_steps=None,
):
    """Plan once, physically allocate every future save before the first update."""
    rank = dist.get_rank()
    steps = checkpoint_steps(initial, total, every, save_steps)
    if not steps:
        return
    sizes = {f"rank-{rank}.pt": MIB} if save_optimizer else {}
    for role, optimizer in optimizers.items():
        network = getattr(model, role)
        states = [(role, lambda: {"model": ModelWrapper(network).state_dict()})]
        if save_optimizer:
            states.append((f"{role}_optimizer", lambda: {"optimizer": optimizer_blueprint(network, optimizer)}))
        for name, make_state in states:
            state = collective_call(make_state)
            capacity, metadata_capacity = plan_capacity(state)
            del state
            sizes[f"{name}/__{rank}_0.distcp"] = capacity
            if rank == 0:
                sizes[f"{name}/.metadata.tmp"] = metadata_capacity
    if rank == 0:
        sizes["complete.json.tmp"] = MIB
    all_sizes = [None] * dist.get_world_size()
    dist.all_gather_object(all_sizes, sizes)
    files = {name: size for part in all_sizes for name, size in part.items()}
    identity = dict(format="wan_control_dcp_reservation_v1", world_size=dist.get_world_size(),
                    torch_version=str(torch.__version__), contract=contract, files=files)
    if rank == 0:
        print(f"CHECKPOINT reserve steps={steps} per_checkpoint_GiB={sum(files.values()) / 2**30:.3f} "
              f"total_GiB={len(steps) * sum(files.values()) / 2**30:.3f}", flush=True)
    for step in steps:
        destination = Path(output) / f"step-{step:07d}"
        slot = Path(output) / ".checkpoint-reservations" / destination.name

        def prepare():
            if rank != 0:
                return
            if destination.exists():
                recover_incomplete(destination, slot, identity)
            slot.mkdir(parents=True, exist_ok=True)
            prepare_manifest(slot, identity)

        collective_call(prepare)

        def allocate():
            for relative, size in sizes.items():
                target = slot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"CHECKPOINT reserve rank={rank} step={step} file={relative} GiB={size / 2**30:.3f}", flush=True)
                status = reserve_output(target, size, legacy_temps=())
                if status == "complete":
                    raise FileExistsError(f"Unexpected finalized file in reservation: {target}")

        collective_call(allocate)
    if rank == 0:
        print("CHECKPOINT reservations ready on all ranks; starting training", flush=True)


class BoundedStream:
    """Refuse any write beyond already allocated space before it reaches disk."""

    def __init__(self, handle, capacity):
        self.handle, self.capacity = handle, capacity

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def write(self, data):
        if self.handle.tell() + memoryview(data).nbytes > self.capacity:
            raise OSError("Checkpoint exceeds preallocated capacity; refusing unreserved write")
        return self.handle.write(data)


class ReservedFileSystem(FileSystem):
    @contextmanager
    def create_stream(self, path, mode):
        if mode != "wb":
            with super().create_stream(path, mode) as stream:
                yield stream
            return
        destination = Path(path)
        reserved = reservation_path(destination)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite saved file {destination}")
        # r+b is essential: wb would truncate and release the reservation.
        with reserved.open("r+b") as handle:
            capacity = os.fstat(handle.fileno()).st_size
            stream = BoundedStream(handle, capacity)
            yield stream
            handle.flush()
            os.fsync(handle.fileno())
            handle.truncate(handle.tell())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(reserved, destination)


def reserved_writer(directory):
    writer = FileSystemWriter(directory, single_file_per_rank=True, thread_count=1)
    writer.fs = ReservedFileSystem()
    return writer


def activate_reservation(directory, contract):
    """Move allocated files to the final checkpoint path without freeing blocks."""
    directory = Path(directory)
    slot = directory.parent / ".checkpoint-reservations" / directory.name
    manifest = json.loads((slot / "reservation.json").read_text())
    if manifest["contract"] != contract or manifest["world_size"] != dist.get_world_size():
        raise ValueError("Checkpoint reservation contract/world size mismatch")
    for relative, size in manifest["files"].items():
        reserved = reservation_path(slot / relative)
        if not reserved.is_file() or reserved.stat().st_size != size:
            raise ValueError(f"Missing/incomplete checkpoint reservation: {reserved}")
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {directory}")
    os.rename(slot, directory)
