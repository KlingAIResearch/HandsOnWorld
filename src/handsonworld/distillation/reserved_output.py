"""Disk-backed output reservations that can be finalized without a second copy."""

import os
from collections.abc import Iterable
from pathlib import Path


_CHUNK_SIZE = 1024 * 1024


def reservation_path(destination: Path) -> Path:
    """Return the ready-reservation path associated with *destination*."""
    return destination.with_name(destination.name + ".reserve")


def _allocating_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".reserve.allocating")


def _write_all_fd(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise OSError("write returned zero bytes")
        view = view[written:]


def reserve_output(
    destination: Path,
    size: int,
    *,
    allocate: bool = True,
    legacy_temps: Iterable[Path] | None = None,
) -> str:
    """Physically reserve *size* bytes beside *destination*.

    An interrupted allocation remains under the ``.reserve.allocating`` suffix
    and is rewritten from byte zero on the next attempt, even when new allocation
    is disabled.  A nonempty legacy rank temporary file is adopted in the same
    way.  Only a fully written, fsynced file is promoted to ``.reserve``.
    """
    destination = Path(destination)
    if destination.exists():
        return "complete"
    if size < 0:
        raise ValueError(f"reservation size must be non-negative, got {size}")

    reservation = reservation_path(destination)
    allocating = _allocating_path(destination)
    legacy_candidates = (
        destination.parent.glob(destination.name + ".rank*.tmp")
        if legacy_temps is None
        else legacy_temps
    )
    nonempty_legacy_temps = []
    for candidate in legacy_candidates:
        path = Path(candidate)
        if path.stat().st_size > 0:
            nonempty_legacy_temps.append(path)
    if len(nonempty_legacy_temps) > 1:
        raise RuntimeError(
            f"multiple nonempty legacy temporary outputs exist for {destination}: "
            f"{nonempty_legacy_temps}"
        )
    if nonempty_legacy_temps and (reservation.exists() or allocating.exists()):
        raise RuntimeError(
            f"legacy temporary output conflicts with an existing reservation for {destination}: "
            f"{nonempty_legacy_temps[0]}"
        )

    if reservation.exists():
        if allocating.exists() and allocating.stat().st_size > 0:
            raise RuntimeError(
                f"ready reservation conflicts with a nonempty partial reservation for {destination}: "
                f"{allocating}"
            )
        actual_size = reservation.stat().st_size
        if actual_size != size:
            raise ValueError(
                f"reservation has size {actual_size}, expected {size}: {reservation}"
            )
        return "reused"

    if nonempty_legacy_temps:
        os.replace(nonempty_legacy_temps[0], allocating)
    elif not allocating.exists() and not allocate:
        return "unreserved"

    fd = os.open(allocating, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        actual_size = os.fstat(fd).st_size
        if actual_size > size:
            raise ValueError(
                f"partial reservation has size {actual_size}, expected at most {size}: {allocating}"
            )

        os.lseek(fd, 0, os.SEEK_SET)
        remaining = size
        while remaining:
            chunk_size = min(remaining, _CHUNK_SIZE)
            _write_all_fd(fd, os.urandom(chunk_size))
            remaining -= chunk_size
        os.fsync(fd)
    finally:
        os.close(fd)

    os.replace(allocating, reservation)
    return "created"


def _write_all_file(file, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = file.write(view)
        if written is None:
            raise OSError("write did not report progress")
        if written == 0:
            raise OSError("write returned zero bytes")
        view = view[written:]


def write_reserved(destination: Path, payload: bytes) -> None:
    """Write *payload* into an exact-size reservation and publish it."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    reservation = reservation_path(destination)
    with open(reservation, "r+b") as file:
        actual_size = os.fstat(file.fileno()).st_size
        expected_size = len(payload)
        if actual_size != expected_size:
            raise ValueError(
                f"reservation has size {actual_size}, payload has size {expected_size}: {reservation}"
            )
        file.seek(0)
        _write_all_file(file, payload)
        file.flush()
        os.fsync(file.fileno())

    os.replace(reservation, destination)


__all__ = ["reservation_path", "reserve_output", "write_reserved"]
