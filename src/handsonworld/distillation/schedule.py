"""Actual RF sigmas shared by the control ODE, KD and DMD entry points."""

import math


def validate_shift(shift):
    if not math.isfinite(shift) or shift < 1:
        raise ValueError("schedule-shift must be finite and >= 1")


def student_sigmas(steps, shift=5.0):
    """Warp an endpoint-inclusive grid once; retain FastGen's 0.999 start.

    Shift 1 explicitly reproduces this entry point's legacy linear schedule.
    For shift > 1, only the pure-noise endpoint is clipped; intermediate values
    follow the reference Wan schedule without rounding to three decimals.
    """
    validate_shift(shift)
    if steps < 1:
        raise ValueError("steps must be positive")
    if shift == 1:
        return [0.999 * (1 - i / steps) for i in range(steps)] + [0.0]
    result = []
    for i in range(steps + 1):
        u = 1 - i / steps
        result.append(min(0.999, shift * u / (1 + (shift - 1) * u)))
    if any(a <= b for a, b in zip(result, result[1:])):
        raise ValueError("schedule-shift/steps produce repeated sigmas at the 0.999 endpoint")
    return result


def euler_sigmas(t_list, substeps, shift=5.0):
    """Densify in pre-shift coordinates while preserving all stored sigmas.

    Explicit t_list entries are actual sigmas, never shifted a second time.
    With the default student grid this reproduces a shifted 48-step grid,
    except for FastGen's initial sigma cap. Shift 1 uses linear sigma substeps.
    """
    validate_shift(shift)
    if substeps < 1:
        raise ValueError("ODE substeps must be positive")
    if (
        len(t_list) < 2 or not all(math.isfinite(t) for t in t_list)
        or t_list[0] > 0.999 or t_list[-1] != 0
        or any(a <= b for a, b in zip(t_list, t_list[1:]))
    ):
        raise ValueError("ODE sigmas must decrease strictly in [0, 0.999] and end at zero")
    positions = [t / (shift - (shift - 1) * t) for t in t_list]
    if shift > 1 and t_list[0] == 0.999:
        positions[0] = 1.0
    result = []
    for index, (hi, lo) in enumerate(zip(positions, positions[1:])):
        result.append(t_list[index])
        for j in range(1, substeps):
            u = hi + (lo - hi) * j / substeps
            result.append(shift * u / (1 + (shift - 1) * u))
    result.append(t_list[-1])
    if any(a <= b for a, b in zip(result, result[1:])):
        raise ValueError("Dense ODE schedule exceeds the 0.999 start; reduce shift or substeps")
    return result
