#!/usr/bin/env python3
"""Evaluate a CSV/JSON of videos with one independent pipeline per GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from handsonworld.inference_batch import (  # noqa: E402
    checkpoint_paths, clip_windows, completed, inference_code_digest, job_signature,
    read_samples, sample_config, shard_samples,
)
from handsonworld.training.config import (  # noqa: E402
    ConfigError, load_config, model_levels, model_spec, resolve_path,
)
from handsonworld.training.metadata import validate_weight_metadata  # noqa: E402
from handsonworld.training.runtime import (  # noqa: E402
    add_backend_to_path, build_inference_pipeline, run_inference, validate_resources,
)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--metadata", "--metadata_path", help="Override inference.metadata from the config")
    result.add_argument("--data-root", "--data_dir")
    result.add_argument("--output-dir", "--save_dir")
    result.add_argument("--output-layout", choices=("flat", "diffsynth"), default=None,
                        help="diffsynth preserves video paths relative to --data-root for legacy GT pairing")
    result.add_argument("--high-checkpoint")
    result.add_argument("--low-checkpoint")
    result.add_argument("--checkpoint", help="Checkpoint override for single-DiT models")
    result.add_argument("--epoch", type=int, help="Zero-based checkpoint epoch; defaults to the final training epoch")
    result.add_argument("--max-samples", "--max_samples", type=positive,
                        help="Global sample limit, applied before GPU sharding")
    result.add_argument("--split", default=None)
    result.add_argument("--num-frames", "--num_frames", type=positive, default=None)
    result.add_argument("--clips-per-video", "--clips_per_video", type=positive, default=None)
    result.add_argument("--num-inference-steps", type=positive, default=None)
    result.add_argument("--height", type=positive, default=None)
    result.add_argument("--width", type=positive, default=None)
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--fps", type=positive, default=None)
    result.add_argument("--text-cfg-scale", type=float, help="Override the configured text CFG scale")
    result.add_argument("--cond-cfg-scale", type=float, help="Override the configured condition CFG scale")
    result.add_argument("--cfg-scale", type=float, help="Broadcast one CFG scale to text and condition")
    result.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--vram-margin-gb", type=float, default=None)
    result.add_argument("--comparison", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--overwrite", action="store_true", help="Regenerate even matching completed outputs")
    result.add_argument("--dry-run", action="store_true", help="Print plan without CUDA, decoding videos or reading weights")
    result.add_argument("--check-inputs", action="store_true", help="Check input paths and trained checkpoint hashes, then exit")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, "infer")
        levels = model_levels(config)
        checkpoints = checkpoint_paths(
            config, args.config, args.high_checkpoint, args.low_checkpoint, args.epoch, args.checkpoint
        )
        inference = config["inference"]
        # CLI values override config values. Config paths belong to the JSON;
        # explicit CLI paths belong to the caller's working directory.
        if args.metadata is None:
            if not inference.get("metadata"):
                raise ConfigError("Set inference.metadata in the config or pass --metadata")
            args.metadata = str(resolve_path(args.config, inference["metadata"]))
        for argument in ("data_root", "output_dir"):
            if getattr(args, argument) is None and inference.get(argument):
                setattr(args, argument, str(resolve_path(args.config, inference[argument])))
        defaults = {
            "split": "all", "num_frames": 81, "clips_per_video": 1,
            "num_inference_steps": 50, "height": 480, "width": 640,
            "seed": 0, "fps": 15, "low_vram": True, "vram_margin_gb": 2.0,
            "comparison": True, "output_layout": "flat", "max_samples": None,
        }
        for key, default in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, inference.get(key, default))
        for level, path in checkpoints.items():
            key = "checkpoint" if level == "single" else f"{level}_checkpoint"
            inference[key] = path
        configured_cfg = inference.get("cfg_scale", [1.0, 2.0])
        if isinstance(configured_cfg, (int, float)):
            configured_cfg = [float(configured_cfg), float(configured_cfg)]
        if not isinstance(configured_cfg, (list, tuple)) or len(configured_cfg) != 2:
            raise ConfigError("config.inference.cfg_scale must contain text and condition scales")
        if args.cfg_scale is not None:
            cfg_scale = [args.cfg_scale, args.cfg_scale]
        else:
            cfg_scale = [
                configured_cfg[0] if args.text_cfg_scale is None else args.text_cfg_scale,
                configured_cfg[1] if args.cond_cfg_scale is None else args.cond_cfg_scale,
            ]
        inference.update(num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
                         height=args.height, width=args.width, seed=args.seed, fps=args.fps,
                         cfg_scale=cfg_scale,
                         low_vram=args.low_vram, vram_margin_gb=args.vram_margin_gb)
        if args.vram_margin_gb < 0:
            raise ConfigError("vram-margin-gb must be non-negative")
        rank, local_rank, world_size = (int(os.environ.get(k, d)) for k, d in
                                       (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
        samples = read_samples(args.metadata, args.data_root, args.split, args.max_samples,
                               output_layout=args.output_layout)
        selected = shard_samples(samples, rank, world_size)
        if not samples:
            raise ConfigError("No eligible samples in the requested metadata/split")
        checkpoint_name = "_".join(
            f"{level}-{Path(checkpoints[level]).stem}" for level in levels
        )
        default_output_root = resolve_path(args.config, config["training"]["output_root"])
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
            default_output_root / "egovid_eval" / checkpoint_name
        )
        print(json.dumps({"rank": rank, "world_size": world_size, "samples_total": len(samples),
                          "samples_this_rank": len(selected), "checkpoints": checkpoints,
                          "metadata": str(Path(args.metadata).expanduser().resolve()),
                          "data_root": str(Path(args.data_root).expanduser().resolve()) if args.data_root else str(Path(args.metadata).expanduser().resolve().parent),
                          "output_dir": str(output_dir), "output_layout": args.output_layout,
                          "model_type": model_spec(config)["model_type"],
                          "representation": config["representation"], "num_frames": args.num_frames,
                          "clips_per_video": args.clips_per_video, "low_vram": args.low_vram,
                          "cfg_scale": inference["cfg_scale"],
                          "switch_dit_boundary": inference.get("switch_dit_boundary")}, indent=2), flush=True)
        if args.dry_run or not selected:
            return 0
        # All assigned files are checked before allocating either 14B expert.
        for sample in selected:
            current = sample_config(config, sample, output_dir, comparison=args.comparison)
            validate_resources(current, args.config, "infer")
        metadata = {level: validate_weight_metadata(
                        path, config["representation"], level,
                        model_type=model_spec(config)["model_type"],
                    )
                    for level, path in checkpoints.items()}
        if args.check_inputs:
            print(f"Checked {len(selected)} samples and all trained checkpoint hashes; no model loaded.")
            return 0

        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise ConfigError("Inference requires CUDA; use --dry-run or --check-inputs on CPU")
        torch.cuda.set_device(local_rank)
        inference["device"] = f"cuda:{local_rank}"
        add_backend_to_path(config, args.config)
        from diffsynth.utils.data import VideoData

        code_digest = inference_code_digest(config, args.config)
        pipe = None
        for sample in selected:
            video = VideoData(sample["video"], height=args.height, width=args.width)
            try:
                lengths = [len(video)]
            finally:
                video.data.reader.close()
            lengths += [len(np.load(sample[key], mmap_mode="r")) for key in ("cam_pose", "hand_pose")]
            if sample.get("camera_intrinsics"):
                intrinsics = np.load(sample["camera_intrinsics"], mmap_mode="r")
                if intrinsics.ndim == 3:
                    lengths.append(len(intrinsics))
            for start, length in clip_windows(min(lengths), args.num_frames, args.clips_per_video):
                current = sample_config(config, sample, output_dir, start, length, args.comparison)
                # GPU assignment is not part of the result: restarting with fewer
                # workers should still recognize the same completed clips.
                signature_config = {**current, "inference": {**current["inference"], "device": "cuda"}}
                signature = job_signature(signature_config, metadata, code_digest)
                if not args.overwrite and completed(current, signature):
                    print(f"Skip completed: {current['inference']['output']}", flush=True)
                    continue
                if pipe is None:
                    for level in levels:
                        validate_weight_metadata(resolve_path(args.config, config["models"][level]["adapter"]),
                                                 config["representation"], level,
                                                 model_type=model_spec(config)["model_type"], allow_initializer=True)
                    pipe = build_inference_pipeline(config, args.config)
                output = run_inference(current, args.config, pipe=pipe)
                record = output.with_suffix(".json")
                temporary = record.with_suffix(f".json.{os.getpid()}.tmp")
                temporary.write_text(json.dumps({"signature": signature, "inference": current["inference"],
                                                 "checkpoints": metadata}, indent=2) + "\n")
                temporary.replace(record)
                print(f"Wrote {output}", flush=True)
        return 0
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        parser().error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
