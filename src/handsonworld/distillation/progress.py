"""Newline progress bars that remain visible through torchrun --tee and log files."""

from collections import deque
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class StepProgress:
    def __init__(self, stage, total, initial=0, steps_per_epoch=None, clock=time.monotonic):
        self.stage = stage.upper()
        self.total = total
        self.iteration = initial
        self.steps_per_epoch = steps_per_epoch
        self.clock = clock
        self.started = self.previous = clock()
        self.step_times = deque(maxlen=20)
        self.wall_times = deque(maxlen=20)

    def update(self, iteration, step_seconds, loss=None, phase=None):
        now = self.clock()
        self.step_times.append(step_seconds)
        self.wall_times.append(now - self.previous)
        self.previous = now
        self.iteration = iteration
        fraction = min(1.0, iteration / self.total)
        filled = int(24 * fraction)
        bar = "#" * filled + "-" * (24 - filled)
        average = sum(self.step_times) / len(self.step_times)
        # Wall intervals include checkpoint pauses; resume measures only this run.
        eta = max(0, self.total - iteration) * sum(self.wall_times) / len(self.wall_times)
        fields = [
            f"PROGRESS {self.stage} [{bar}] {iteration}/{self.total} ({100 * fraction:.1f}%)",
            f"step={step_seconds:.1f}s avg20={average:.1f}s",
            f"elapsed={duration(now - self.started)} ETA={duration(eta)}",
        ]
        if self.steps_per_epoch:
            fields.append(f"epoch={iteration / self.steps_per_epoch:.2f}")
        if loss is not None:
            fields.append(f"loss(rank0)={loss:.5g}")
        if phase:
            fields.append(f"phase={phase}")
        return " | ".join(fields)


def global_step_seconds(started, device):
    """Report the slowest rank, including fetching data and condition encoding."""
    import torch
    import torch.distributed as dist

    elapsed = torch.tensor(time.monotonic() - started, device=device, dtype=torch.float64)
    dist.reduce(elapsed, dst=0, op=dist.ReduceOp.MAX)
    return elapsed.item() if dist.get_rank() == 0 else None
