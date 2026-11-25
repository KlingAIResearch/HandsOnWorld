"""Node-level coordination before/after Accelerate, using a shared directory."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time
from collections.abc import Callable

from .config import ConfigError


def cluster_arguments(args) -> list[str] | None:
    if args.num_machines is None:
        if any(value is not None for value in (args.machine_rank, args.num_processes, args.main_process_ip, args.launch_id)):
            raise ConfigError("Cluster options require --num-machines")
        return None
    if args.accelerate_config is not None:
        raise ConfigError("Use either explicit cluster topology or --accelerate-config")
    if args.num_machines < 1 or args.machine_rank is None or not 0 <= args.machine_rank < args.num_machines:
        raise ConfigError("machine-rank must be in [0, num-machines)")
    if args.num_processes is None or args.num_processes < args.num_machines or args.num_processes % args.num_machines:
        raise ConfigError("num-processes must be positive and divisible by num-machines")
    if not args.main_process_ip or not 1 <= args.main_process_port <= 65535:
        raise ConfigError("Cluster launch requires a master address and valid port")
    if not math.isfinite(args.rendezvous_timeout) or args.rendezvous_timeout <= 0:
        raise ConfigError("rendezvous-timeout must be finite and positive")
    if args.num_machines > 1:
        if not args.rendezvous_dir or not args.launch_id or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", args.launch_id):
            raise ConfigError("Multi-node launch requires rendezvous-dir and a safe, unique launch-id")
    options = [
        "--num_machines", str(args.num_machines),
        "--machine_rank", str(args.machine_rank),
        "--num_processes", str(args.num_processes),
        "--main_process_ip", args.main_process_ip,
        "--main_process_port", str(args.main_process_port),
        "--mixed_precision", args.mixed_precision,
        "--dynamo_backend", "no",
    ]
    if args.num_processes > 1:
        options += ["--multi_gpu", "--same_network"]
    return options


class NodeCoordinator:
    """Run local work, then ensure every node succeeded before the next phase."""

    def __init__(self, directory: Path, launch_id: str, rank: int, nodes: int, timeout: float, contract: str):
        self.directory = directory / launch_id
        self.rank, self.nodes, self.timeout, self.contract = rank, nodes, timeout, contract

    def run(self, stage: str, action: Callable[[], None], *, timeout: float | None = None) -> None:
        directory = self.directory / stage
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"node-{self.rank}.json"
        payload = {"contract": self.contract, "error": None}
        error = None
        try:
            action()
        except Exception as exc:
            error = exc
            payload["error"] = f"{type(exc).__name__}: {exc}"
        def publish():
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, path)

        publish()
        if error is not None:
            raise error
        wait_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + wait_timeout
        try:
            while True:
                pending = []
                for rank in range(self.nodes):
                    peer = directory / f"node-{rank}.json"
                    if not peer.is_file():
                        pending.append(rank)
                        continue
                    status = json.loads(peer.read_text(encoding="utf-8"))
                    if status.get("contract") != self.contract:
                        raise ConfigError(f"Node {rank} has a different training configuration in {stage}")
                    if status.get("error"):
                        raise ConfigError(f"Node {rank} failed during {stage}: {status['error']}")
                if not pending:
                    return
                if time.monotonic() >= deadline:
                    raise ConfigError(f"Timed out during {stage}; waiting for nodes {pending}")
                time.sleep(min(0.1, wait_timeout))
        except Exception as exc:
            # A late peer must see a timeout/failure, not an abandoned ready marker.
            payload["error"] = f"{type(exc).__name__}: {exc}"
            publish()
            raise
