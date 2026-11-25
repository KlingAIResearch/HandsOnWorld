"""Hand-surface Plücker attributes used by the pinned training backend.

All vectors are in the same world coordinate frame. The caller rasterizes
vertex directions and positions before calling ``plucker_from_direction``.
These functions preserve the original reflection/normal formulas and eps.
"""

import torch
import torch.nn.functional as F

CONTROL_MODES = ("normal_plucker", "reflection_plucker")


def surface_directions(normals, view_directions, mode):
    """Return vertex attributes before interpolation; normals must be unit vectors.

    ``view_directions`` uses the backend's camera-to-surface convention.
    A reflection ray depends on the viewing direction; it is not equivalent to
    the camera-independent normal ray described by the normal representation.
    """
    if mode == "normal_plucker":
        return normals
    if mode == "reflection_plucker":
        reflected = view_directions - 2.0 * (view_directions * normals).sum(dim=-1, keepdim=True) * normals
        return F.normalize(reflected, dim=-1, eps=1e-6)
    raise ValueError(f"Unsupported control mode {mode!r}; expected one of {CONTROL_MODES}")


def plucker_from_direction(positions, directions):
    """Construct [unit direction, position x unit direction] after interpolation."""
    direction = F.normalize(directions, dim=-1, eps=1e-6)
    return torch.cat([direction, torch.cross(positions, direction, dim=-1)], dim=-1)
