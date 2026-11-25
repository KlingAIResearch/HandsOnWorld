"""Prepare configured training assets: local reuse before public model downloads."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import uuid

from .config import DEFAULT_MODEL_TYPE, MODEL_ALIASES, ConfigError, model_levels, model_spec, resolve_path
from .metadata import metadata_path, write_initializer_metadata

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
TOKENIZER_MODEL_ID = "google/umt5-xxl"
TOKENIZER_FILES = ("spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
ADAPTER_DIM = 5120
PACKED_INPUT_DIM = 48 * 8 * 8
REPO_ROOT = Path(__file__).resolve().parents[3]
INITIALIZER_MODEL_HASHES = {
    DEFAULT_MODEL_TYPE: "fef26cc2ee3a832746d9af9cda5f489f",
    "wan2.1-i2v-14b-480p": "889b280d659e713c76747531807487ee",
    "wan2.2-ti2v-5b": "d9e385713c5a0eafa97b2f5a8a833815",
}
RESOURCE_ASSET_KEYS = frozenset({
    "faces", "metadata", "text_encoder", "vae", "image_encoder", "tokenizer",
})


@contextmanager
def preparation_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".resource-prepare.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _present(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_symlink() or path.exists():
        raise ConfigError(f"Expected a resource file (or found a broken link): {path}")
    return False


def link_resource(source: Path, target: Path) -> None:
    if _present(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())
    print(f"Reuse: {target} -> {source}", flush=True)


def verify_neutral_initializer(
    path: Path, *, model_type: str = DEFAULT_MODEL_TYPE,
    dim: int | None = None, downscale_factor: int | None = None,
) -> None:
    """Reject trained, truncated or differently shaped legacy control checkpoints."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ConfigError(f"Invalid safetensors initializer: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= 100_000_000:
            raise ConfigError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(size))
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if any("lora_" in key for key in tensors):
            raise ConfigError(f"Refusing to relabel a trained LoRA checkpoint as initialization: {path}")
        if dim is None:
            dim = ADAPTER_DIM if model_type == DEFAULT_MODEL_TYPE else model_spec({"model_type": model_type})["dim"]
        if downscale_factor is None:
            downscale_factor = 8 if model_type == DEFAULT_MODEL_TYPE else model_spec({"model_type": model_type})["downscale_factor"]
        packed_input_dim = PACKED_INPUT_DIM if model_type == DEFAULT_MODEL_TYPE else 48 * downscale_factor**2
        expected = {
            "control_adapter.conv.weight": [dim, packed_input_dim, 2, 2],
            "control_adapter.conv.bias": [dim],
            "control_adapter.residual_blocks.0.conv1.weight": [dim, dim, 3, 3],
            "control_adapter.residual_blocks.0.conv1.bias": [dim],
            "control_adapter.residual_blocks.0.conv2.weight": [dim, dim, 3, 3],
            "control_adapter.residual_blocks.0.conv2.bias": [dim],
        }
        actual_keys = {key for key in tensors if key.startswith("control_adapter.")}
        if actual_keys != set(expected):
            raise ConfigError(f"Incomplete or unexpected 48-channel control adapter: {path}")
        data_size = path.stat().st_size - 8 - size
        for key, shape in expected.items():
            entry = tensors[key]
            if entry.get("shape") != shape or entry.get("dtype") not in {"BF16", "F16", "F32"}:
                raise ConfigError(f"Incompatible initializer tensor {key}: {path}")
            begin, end = entry["data_offsets"]
            elements = 1
            for dimension in shape:
                elements *= dimension
            if not 0 <= begin <= end <= data_size or end - begin != elements * (4 if entry["dtype"] == "F32" else 2):
                raise ConfigError(f"Invalid initializer tensor extent {key}: {path}")
            if ".conv1." in key:
                continue
            stream.seek(8 + size + begin)
            remaining = end - begin
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk or chunk.strip(b"\x00"):
                    raise ConfigError(f"Control tensor is not zero-initialized: {key} in {path}")
                remaining -= len(chunk)
        # Shape hashing is the same contract used by the vendored DiffSynth
        # registry. It distinguishes Wan2.1 I2V from Wan2.2 A14B even though
        # both control adapters have the same 5120-wide shape.
        base_tensors = {key: value for key, value in tensors.items() if not key.startswith("control_adapter.")}
        if base_tensors:
            signatures = []
            for key, entry in tensors.items():
                shape = "_".join(str(value) for value in entry["shape"])
                signatures.extend((f"{key}:{shape}", key))
            actual_hash = hashlib.md5(",".join(sorted(signatures)).encode("UTF-8")).hexdigest()
            expected_hash = INITIALIZER_MODEL_HASHES[model_type]
            if actual_hash != expected_hash:
                raise ConfigError(
                    f"Initializer backbone does not match {model_type}: {path} "
                    f"(shape hash {actual_hash}, expected {expected_hash})"
                )


def prepare_resources(config: dict, config_path: str | Path, *, levels=None,
                      offline: bool = False, reuse_roots=None, asset_keys=None, seed: int = 0) -> dict:
    """Only public Wan assets may be downloaded; private datasets/faces need local sources."""
    if asset_keys is None:
        selected_assets = RESOURCE_ASSET_KEYS
    else:
        if isinstance(asset_keys, str):
            raise ConfigError("asset_keys must be an iterable of asset names, not a string")
        try:
            selected_assets = frozenset(asset_keys)
        except TypeError as error:
            raise ConfigError("asset_keys must be an iterable of asset names") from error
        unknown_assets = selected_assets - RESOURCE_ASSET_KEYS
        if unknown_assets:
            raise ConfigError(
                f"Unknown resource asset keys: {sorted(unknown_assets)}; "
                f"expected a subset of {sorted(RESOURCE_ASSET_KEYS)}"
            )
    settings = config.get("resources", {})
    revision = settings.get("revision", "main")
    spec = model_spec(config)
    if levels is None:
        levels = model_levels(config)
    invalid_levels = set(levels) - set(spec["levels"])
    if invalid_levels:
        raise ConfigError(f"Invalid model levels for {spec['model_type']}: {sorted(invalid_levels)}")
    roots = [Path(root).expanduser().resolve() for root in (reuse_roots if reuse_roots is not None else [REPO_ROOT.parent])]
    model_name = spec["model_id"].rsplit("/", 1)[-1]
    family = model_name.split("-", 1)[0]
    model_bases = []
    for root in roots:
        model_bases.extend((root / family / model_name,
                            root / "DiffSynth-Studio/models/Wan-AI" / model_name,
                            root / "models/Wan-AI" / model_name,
                            root / model_name, root))
    tokenizer_bases = []
    for root in roots:
        tokenizer_bases.extend((root / "Wan2.2/Wan2.2-I2V-A14B",
                                root / "DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B",
                                root / "models/Wan-AI/Wan2.2-I2V-A14B",
                                root / "Wan2.2-I2V-A14B", *model_bases))
    actions = {"reused": [], "downloaded": [], "initialized": []}

    def reuse(target, candidates):
        if _present(target):
            return True
        for source in candidates:
            if source.is_file():
                link_resource(source, target)
                actions["reused"].append(str(target))
                return True
        return False

    def public_file(target, relative, *, repo_id=None, bases=None,
                    download_relative=None, download_revision=None):
        if reuse(target, [base / relative for base in (bases or model_bases)]):
            return
        if offline:
            raise FileNotFoundError(f"Missing public model file in offline mode: {target} ({relative})")
        from huggingface_hub import hf_hub_download
        repo_id = repo_id or spec["model_id"]
        download_relative = download_relative or relative
        download_revision = download_revision or revision
        print(
            f"Download missing model file: {repo_id}/{download_relative} "
            f"(revision={download_revision})", flush=True,
        )
        cached = hf_hub_download(
            repo_id=repo_id, filename=download_relative, revision=download_revision
        )
        link_resource(Path(cached), target)
        actions["downloaded"].append(relative)

    with preparation_lock(REPO_ROOT / "outputs"):
        # Fail on unavailable private inputs before downloading any large weights.
        if "faces" in selected_assets:
            faces = resolve_path(config_path, config["hand_mesh_faces_path"])
            face_sources = [candidate for root in roots for candidate in (
                root / "HaWoR/hand_mesh_faces.txt", root / "hand_mesh_faces.txt",
            )]
            if not reuse(faces, face_sources if faces.name == "hand_mesh_faces.txt" else []):
                raise FileNotFoundError(
                    f"Missing hand mesh faces: {faces}. Supply a local faces file; "
                    "this is not a public Wan model asset."
                )
        if "metadata" in selected_assets:
            metadata = resolve_path(config_path, config["data"]["metadata"])
            if not metadata.is_file():
                raise FileNotFoundError(
                    f"Missing dataset metadata: {metadata}. Configure data.metadata with an "
                    "existing training manifest, or create one with scripts/prepare_arctic.py "
                    "or scripts/prepare_videos.py."
                )
        models = config["models"]
        if "text_encoder" in selected_assets:
            public_file(
                resolve_path(config_path, models["text_encoder"]),
                "models_t5_umt5-xxl-enc-bf16.pth",
            )
        if "vae" in selected_assets:
            public_file(resolve_path(config_path, models["vae"]), spec["vae_filename"])
        if "image_encoder" in selected_assets and spec["image_encoder_filename"]:
            public_file(resolve_path(config_path, models["image_encoder"]), spec["image_encoder_filename"])
        if "tokenizer" in selected_assets:
            tokenizer = resolve_path(config_path, models["tokenizer"])
            for name in TOKENIZER_FILES:
                tokenizer_options = {}
                if spec["model_type"] != DEFAULT_MODEL_TYPE:
                    tokenizer_options = {
                        "repo_id": TOKENIZER_MODEL_ID,
                        "download_relative": name,
                        "download_revision": settings.get("tokenizer_revision", "main"),
                    }
                public_file(
                    tokenizer / name, f"google/umt5-xxl/{name}",
                    bases=tokenizer_bases, **tokenizer_options,
                )
        for level in levels:
            target = resolve_path(config_path, models[level]["adapter"])
            filename = spec["initializer_filenames"][level]
            alias = next(name for name, model_type in MODEL_ALIASES.items() if model_type == spec["model_type"])
            initialized_root = resolve_path(config_path, "../models/initialized")
            legacy_roots = [initialized_root]
            legacy_roots.extend(root / "models/initialized" for root in roots)
            candidates = [base / filename for base in model_bases]
            for root in legacy_roots:
                for representation in ("normal_plucker", "reflection_plucker"):
                    directory = root / representation if len(spec["levels"]) == 2 else root / alias / representation
                    candidates.append(directory / f"{level}.safetensors")
            source = next((p for p in candidates if p.is_file() and p.absolute() != target.absolute()), None)
            if _present(target):
                # Recover an interrupted legacy import only when its known source
                # and neutral control weights can be verified again.
                if (not metadata_path(target).exists() and target.is_symlink()
                        and source is not None and target.resolve() == source.resolve()):
                    verify_neutral_initializer(source, model_type=spec["model_type"])
                    write_initializer_metadata(target, level,
                                   model_type=spec["model_type"],
                                   kind="zero_initialized_control_adapter", source_name=source.name,
                                   zero_initialization_verified=True)
                continue  # Never relabel arbitrary existing or mismatched checkpoints.
            if source is not None:
                print(f"Verify zero-initialized 48-channel adapter: {source}", flush=True)
                verify_neutral_initializer(source, model_type=spec["model_type"])
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.safetensors")
                try:
                    temporary.symlink_to(source.resolve())
                    print(f"Hash initializer and write metadata: {target}", flush=True)
                    write_initializer_metadata(temporary, level,
                                   model_type=spec["model_type"],
                                   kind="zero_initialized_control_adapter", source_name=source.name,
                                   zero_initialization_verified=True)
                    # Publish metadata first; a weight alias is never exposed without it.
                    os.replace(metadata_path(temporary), metadata_path(target))
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                    metadata_path(temporary).unlink(missing_ok=True)
                print(f"Reuse initializer: {target} -> {source}", flush=True)
                actions["reused"].append(str(target))
                continue
            shards = models[level].get("source_shards")
            if not shards:
                raise ConfigError(f"Missing models.{level}.source_shards for adapter initialization")
            paths = [resolve_path(config_path, value) for value in shards]
            for path in paths:
                relative = f"{level}_noise_model/{path.name}" if len(spec["levels"]) == 2 else path.name
                public_file(path, relative)
            from .runtime import add_backend_to_path, initialize_adapter
            add_backend_to_path(config, config_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.safetensors")
            try:
                print(f"Initialize {level} control adapter from local base shards: {target}", flush=True)
                initializer_options = {"seed": seed, "in_dim": 48}
                if spec["model_type"] != DEFAULT_MODEL_TYPE:
                    initializer_options.update(
                        model_dim=spec["dim"],
                        downscale_factor=spec["downscale_factor"],
                    )
                initialize_adapter(paths, temporary, **initializer_options)
                write_initializer_metadata(temporary, level,
                               model_type=spec["model_type"],
                               initialization_seed=seed,
                               base_model_id=spec["model_id"], revision=revision)
                os.replace(metadata_path(temporary), metadata_path(target))
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
                metadata_path(temporary).unlink(missing_ok=True)
            actions["initialized"].append(str(target))
    return actions
