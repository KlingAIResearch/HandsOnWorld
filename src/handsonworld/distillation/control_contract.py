"""Control representation and explicit cross-teacher KD provenance checks."""

import json
from pathlib import Path


REPRESENTATIONS = ("reflection_plucker", "normal_plucker")


def control_representation(contract):
    representation = contract.get(
        "control_representation",
        contract.get("raw_control_settings", {}).get("render_mode", "reflection_plucker"),
    )
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported control representation: {representation}")
    return representation


def validate_control_metadata(path, representation, digest, model_type=None):
    """Validate normal weights; allow sidecar-less legacy reflection files."""
    sidecar = Path(str(path) + ".handsonworld.json")
    if not sidecar.is_file():
        if representation == "normal_plucker":
            raise ValueError(f"normal_plucker requires its HandsOnWorld metadata sidecar: {sidecar}")
        return
    metadata = json.loads(sidecar.read_text())
    expected = {
        "representation": representation,
        "model_type": model_type or "wan2.1-i2v-14b-480p",
        "noise_level": "single",
        "control_in_dim": 48,
        "attr_space": "world",
        "channel_order": ["hand_plucker_world", "camera_plucker"],
        "weight_sha256": digest,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Control checkpoint metadata mismatch: {key}")


def validate_ode_control(manifest, contract, *, allow_cross_control=False):
    """Permit a different teacher only for compact KD with raw geometry lookup.

    Cached adapter features/packed maps must never be reinterpreted as a new
    representation. The compact loader returns raw poses via source_index, so
    its student condition can be rendered and encoded independently.
    """
    if manifest.get("format") == "handsonworld_ode_trajectory_v1":
        if contract["stage"] != "kd":
            raise ValueError("Public ODE trajectories are only valid for KD")
        source_representation = manifest.get("control_representation")
        target_representation = manifest.get("target_control_representation", source_representation)
        if target_representation != control_representation(contract):
            raise ValueError("ODE target control representation differs from the KD recipe")
        cross_control = source_representation != target_representation
        if cross_control and not (allow_cross_control and manifest.get("allow_cross_control") is True):
            raise ValueError("Cross-control ODE requires both manifest provenance and explicit CLI opt-in")
        if allow_cross_control and not cross_control:
            raise ValueError("--allow-cross-control-ode was set but the ODE source and target controls are identical")
        return {
            "control_representation": source_representation,
            "student_checkpoint_digest": manifest.get("source_student_checkpoint_digest"),
        } if cross_control else None

    source = manifest.get("provenance", {})
    source_digest = source.get("control_sha256")
    compact = manifest.get("format") == "wan_i2v_control_ode_v1"
    if allow_cross_control and (contract["stage"] != "kd" or not compact):
        raise ValueError("--allow-cross-control-ode requires KD compact ODE data with raw conditions")
    different = (
        source_digest is not None and source_digest != contract["control_sha256"]
    ) or control_representation(source) != control_representation(contract)
    if different and not allow_cross_control:
        raise ValueError("ODE/cache control differs from student; compact KD requires --allow-cross-control-ode")
    if not allow_cross_control:
        return None
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise ValueError("Cross-control ODE requires the source teacher control SHA-256")
    # Changing control does not authorize a different base architecture, video
    # layout or coordinate convention. Timesteps and source fingerprints are
    # checked independently by the ODE dataset, as in ordinary KD.
    for key in ("frames", "height", "width", "layers"):
        if source.get(key) != contract[key]:
            raise ValueError(f"Cross-control ODE mismatch: {key}")
    if Path(source.get("weights", "")).resolve() != Path(contract["weights"]).resolve():
        raise ValueError("Cross-control ODE mismatch: weights")
    settings = source.get("raw_control_settings", {})
    if settings.get("attr_space") != "world" or settings.get("intrinsics_space") != "target":
        raise ValueError("Cross-control ODE requires world attributes and target-space intrinsics")
    return {
        "control_sha256": source_digest,
        "control_representation": control_representation(source),
        "teacher_checkpoint": manifest.get("teacher_checkpoint"),
    }
