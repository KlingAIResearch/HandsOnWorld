"""Export only the causal student and frozen adapter, on CPU, for publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile


BUNDLE_FORMAT = "handsonworld_causal_student_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_student(checkpoint, control_checkpoint, output, *, dtype="bfloat16"):
    """Gather DCP shards into CPU tensors without materializing a Wan model.

    Requires host RAM for a full student and serialization buffers. No optimizer,
    teacher or fake-score weights are read. The published contract omits local
    training paths; this is an inference bundle, not an exact-resume checkpoint.
    """
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint, control_checkpoint, output = map(Path, (checkpoint, control_checkpoint, output))
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("Export dtype must be bfloat16 or float32")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite export: {output}")
    marker = checkpoint / "complete.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete DCP checkpoint: {checkpoint}")
    source = json.loads(marker.read_text())
    contract = dict(source["contract"])
    if contract.get("control_sha256") != sha256(control_checkpoint):
        raise ValueError("Control checkpoint differs from the student training contract")
    if not contract.get("model_type"):
        # Historical FastGen checkpoints predate the explicit backbone field.
        # Recover it only from metadata bound to the verified teacher weights.
        sidecar = Path(str(control_checkpoint) + ".handsonworld.json")
        if not sidecar.is_file():
            raise ValueError("Legacy student has no model_type; a verified control sidecar is required")
        control_metadata = json.loads(sidecar.read_text())
        if control_metadata.get("weight_sha256") != contract["control_sha256"]:
            raise ValueError("Control sidecar does not match the verified teacher checkpoint")
        from handsonworld.models.specs import DISTILLATION_MODEL_TYPES, get_model_spec

        model_type = control_metadata.get("model_type")
        if model_type not in DISTILLATION_MODEL_TYPES:
            raise ValueError("Control sidecar has no supported distillation model_type")
        if contract.get("layers") != get_model_spec(model_type)["num_layers"]:
            raise ValueError("Legacy student layers do not match the control sidecar backbone")
        contract["model_type"] = model_type
    reader = dcp.FileSystemReader(checkpoint / "net")
    metadata = reader.read_metadata()
    state = {}
    for name, value in metadata.state_dict_metadata.items():
        if not name.startswith("model.") or not isinstance(value, TensorStorageMetadata):
            raise ValueError(f"Unexpected non-model DCP entry: {name}")
        tensor_dtype = getattr(torch, dtype) if value.properties.dtype.is_floating_point else value.properties.dtype
        state[name.removeprefix("model.")] = torch.empty(value.size, dtype=tensor_dtype, device="cpu")
    if not state:
        raise ValueError("Student checkpoint contains no tensors")
    dcp.load({"model": state}, storage_reader=reader)
    with safe_open(control_checkpoint, framework="pt", device="cpu") as handle:
        adapter = {key: handle.get_tensor(key).to(getattr(torch, dtype)) for key in handle.keys()
                   if key.startswith("dit.control_adapter.")}
    if not adapter:
        raise ValueError("Control checkpoint contains no adapter tensors")
    public_keys = (
        "stage", "model_type", "control_representation", "control_sha256", "chunk_size",
        "fixed_first_frame", "steps", "t_list", "schedule_shift", "height", "width", "frames",
    )
    public_contract = {key: contract[key] for key in public_keys if key in contract}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        save_file(state, staging / "student.safetensors")
        save_file(adapter, staging / "adapter.safetensors")
        record = {
            "format": BUNDLE_FORMAT, "iteration": source["iteration"], "contract": public_contract,
            "source_control_sha256": contract["control_sha256"], "dtype": dtype,
            "student_sha256": sha256(staging / "student.safetensors"),
            "adapter_sha256": sha256(staging / "adapter.safetensors"),
            "training_state_complete": False, "weights_only": True,
        }
        (staging / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
        staging.rename(output)
    return output
