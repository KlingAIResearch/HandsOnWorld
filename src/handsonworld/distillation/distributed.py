"""Node-local FSDP2 sharding and replica-aware gradient clipping."""

import os
import socket

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def validate_node_layout(ranks):
    """Require homogeneous torchrun nodes with contiguous ranks per host."""
    local_size = ranks[0]["local_world_size"]
    if local_size < 1 or len(ranks) % local_size:
        raise ValueError("World size must be divisible by LOCAL_WORLD_SIZE")
    hosts = []
    for start in range(0, len(ranks), local_size):
        node = ranks[start:start + local_size]
        host = node[0]["hostname"]
        if any(
            item["local_world_size"] != local_size
            or item["local_rank"] != offset
            or item["hostname"] != host
            for offset, item in enumerate(node)
        ):
            raise ValueError("Node-local sharding requires equal-size, contiguous torchrun ranks per host")
        if host in hosts:
            raise ValueError("Multiple torchrun node groups report the same host; check node ranks")
        hosts.append(host)
    return local_size


def build_training_mesh(strategy):
    world = dist.get_world_size()
    if strategy == "full":
        shape, names = (world,), ("shard",)
    else:
        local = {
            "hostname": socket.gethostname(),
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
        }
        ranks = [None] * world
        dist.all_gather_object(ranks, local)
        local_size = validate_node_layout(ranks)
        replicas = world // local_size
        shape, names = ((replicas, local_size), ("replicate", "shard")) if replicas > 1 else (
            (local_size,), ("shard",)
        )
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)
    return mesh


def clip_grad(optimizers, max_norm, shard_group=None, device="cuda"):
    # Replicated gradients have already been averaged by HSDP. Sum squares
    # across a single shard group so the replica count does not inflate norms.
    total = torch.zeros((), device=device, dtype=torch.float64)
    gradients = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                gradients.append(grad)
                for chunk in grad.detach().reshape(-1).split(1_048_576):
                    total += chunk.double().square().sum().to(total.device)
    dist.all_reduce(total, group=shard_group)
    norm = total.sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError(f"Nonfinite global gradient norm: {norm.item()}")
    # CFG=1 and identical initial teacher/fake-score weights give a legitimate
    # zero student gradient. Match clip_grad_norm_: zero is finite, not failure.
    factor = min(1.0, max_norm / (norm.item() + 1e-6)) if max_norm > 0 else 1.0
    for grad in gradients:
        grad.mul_(factor)
    return norm.item()
