"""HandsOnWorld causal SFT, knowledge distillation and Self-Forcing.

Run via torchrun. Real training consumes explicitly cached, aligned conditions;
--synthetic-data is solely a capacity/gradient smoke test.
"""

# ruff: noqa: E402 -- script configures its repository import path before imports.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path
import time
from datetime import timedelta
from types import SimpleNamespace

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from fastgen.methods import CausalSFTModel, CausalKDModel, SelfForcingModel
from fastgen.methods.distribution_matching.causvid import CausVidModel
from fastgen.configs.opt import get_optimizer
from fastgen.utils import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaLinearScheduler
from handsonworld.models.specs import DISTILLATION_MODEL_TYPES, get_model_spec
from handsonworld.models.wan import WanI2V, CausalWanI2V
from handsonworld.models.checkpoints import (
    DiffSynthControlCheckpoint,
    encode_control,
    stream_wan_official_local_state,
)
from handsonworld.distillation.checkpoint import (
    save_checkpoint, resume_checkpoint, load_student, load_unsharded_student,
    load_safetensors_student,
)
from handsonworld.distillation.checkpoint_schedule import checkpoint_steps
from handsonworld.distillation.distributed import build_training_mesh, clip_grad
from handsonworld.distillation.data_sampler import ResumeDistributedSampler
from handsonworld.distillation.local_errors import record_local_errors
from fastgen.networks.causal_utils import temporal_chunk_ranges
from handsonworld.distillation.progress import StepProgress, global_step_seconds
from handsonworld.distillation.schedule import student_sigmas, euler_sigmas, validate_shift
from handsonworld.distillation.control_contract import (
    REPRESENTATIONS, control_representation, validate_control_metadata, validate_ode_control,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("sft", "kd", "dmd", "ode"), required=True)
    p.add_argument("--model-type", choices=DISTILLATION_MODEL_TYPES, required=True)
    p.add_argument("--weights", type=Path, required=True, help="Base Wan checkpoint directory or file")
    p.add_argument("--control-checkpoint", type=Path, required=True, help="DiffSynth LoRA + adapter checkpoint")
    p.add_argument("--control-representation", choices=REPRESENTATIONS, default="reflection_plucker")
    p.add_argument("--allow-cross-control-ode", action="store_true",
                   help="KD only: reuse another control teacher's compact ODE targets with newly rendered conditions")
    p.add_argument("--data", type=Path, help="Cached-data directory or index.json")
    p.add_argument("--raw-lmdb", type=Path, nargs="+", help="Existing Wan2.1 I2V raw-condition LMDBs")
    p.add_argument("--text-cache", type=Path, help="Precomputed T5 embeddings for --raw-lmdb")
    p.add_argument("--diffsynth-root", type=Path, help="Legacy raw-condition renderer backend")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--student-init", type=Path, help="Completed SFT/KD checkpoint directory; load only net")
    p.add_argument("--resume", type=Path, help="Completed same-stage checkpoint; restore Adam, scheduler, RNG")
    p.add_argument("--synthetic-data", action="store_true", help="Explicit smoke test, NOT real training")
    p.add_argument(
        "--smoke-test", action="store_true", help="Allow truncated depth with real cached data for integration tests"
    )
    p.add_argument("--layers", type=int, help="Override model depth; smoke tests only")
    p.add_argument("--video-frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--chunk-size", type=int, default=None, help="Latent frames per generated chunk; default 4 with fixed prefix, otherwise 3")
    p.add_argument(
        "--fixed-first-frame", action=argparse.BooleanOptionalAction, default=True,
        help="Keep first I2V latent clean at t=0, exclude its loss, and cache it before generation",
    )
    p.add_argument("--steps", type=int, default=4, help="Few-step student denoising steps")
    p.add_argument(
        "--schedule-shift", type=float, default=5.0,
        help="ODE/KD/DMD sigma grid shift and DMD noise-sampling shift; 1 selects the legacy linear grid; SFT unchanged",
    )
    p.add_argument(
        "--t-list", type=float, nargs="+", default=None,
        help="Actual RF sigmas including final zero, already shifted; overrides student grid; KD must match data",
    )
    p.add_argument("--iterations", type=int, default=3000, help="Total optimizer updates, including resumed updates")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-steps", type=int, nargs="+", default=None,
                   help="Explicit checkpoint milestones, replacing save-every; final step is always saved")
    p.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=True,
                   help="Save Adam/RNG/scheduler for exact resume; disabling keeps only trainable model weights")
    p.add_argument("--preallocate-checkpoints", action=argparse.BooleanOptionalAction, default=None,
                   help="Physically reserve all future checkpoint files before training; enabled by default for KD")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="Rank-zero step progress, timing and ETA; --no-progress keeps JSON diagnostics only")
    p.add_argument("--lr", type=float, default=None,
                   help="Student learning rate: DMD 2e-6, SFT/KD 1e-5")
    p.add_argument("--fake-score-lr", type=float, default=4e-7,
                   help="Independent DMD fake-score learning rate")
    p.add_argument("--adam-betas", type=float, nargs=2, default=None,
                   help="Both optimizers: DMD defaults to 0 0.999; other stages 0.9 0.999")
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--student-update-freq", type=int, default=None,
                   help="Exclusive update cycle: DMD default 6 means one student and five fake-score updates")
    p.add_argument(
        "--streaming-student-backward",
        action="store_true",
        help="DMD: replay all exits and backward one chunk at a time",
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--storage-dtype", choices=("bfloat16", "float32"), default=None,
        help="Parameter/gradient/Adam storage: SFT/DMD default to float32; other stages to bfloat16",
    )
    p.add_argument(
        "--sharding-strategy", choices=("node", "full"), default="node",
        help="node: shard within each host and replicate across hosts; full: shard across all ranks",
    )
    p.add_argument("--lora-scale", type=float, default=1.0, help="Already includes alpha/r; training script alpha/r=1")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-offload-roles", nargs="*", choices=("net", "teacher", "fake_score"), default=[])
    p.add_argument(
        "--adapter-cpu-offload",
        action="store_true",
        help="Move the frozen adapter to CPU after each batch; default keeps it on GPU",
    )
    p.add_argument("--ode-substeps", type=int, default=12,
                   help="Euler substeps per stored interval, spaced before applying schedule-shift")
    p.add_argument("--ode-samples", type=int, default=8, help="Number of global generated trajectories")
    args = p.parse_args()
    if args.lr is None:
        args.lr = 2e-6 if args.stage == "dmd" else 1e-5
    if args.adam_betas is None:
        args.adam_betas = [0.0, 0.999] if args.stage == "dmd" else [0.9, 0.999]
    if args.student_update_freq is None:
        args.student_update_freq = 6 if args.stage == "dmd" else 5
    if args.save_steps is None and args.stage == "dmd":
        args.save_steps = [500, 1000, 2000, 3000]
    if args.storage_dtype is None:
        args.storage_dtype = "float32" if args.stage in ("sft", "dmd") else "bfloat16"
    if not all(math.isfinite(lr) and lr > 0 for lr in (args.lr, args.fake_score_lr)):
        p.error("Learning rates must be finite and positive")
    if not all(math.isfinite(beta) and 0 <= beta < 1 for beta in args.adam_betas):
        p.error("Adam betas must be finite and in [0, 1)")
    try:
        checkpoint_steps(0, args.iterations, args.save_every, args.save_steps)
    except ValueError as error:
        p.error(str(error))
    if args.chunk_size is None:
        args.chunk_size = 4 if args.fixed_first_frame else 3
    spec = get_model_spec(args.model_type)
    if args.layers is None:
        args.layers = spec["num_layers"]
    args.frames = (args.video_frames - 1) // spec["temporal_downscale"] + 1
    if args.video_frames < 1 or (args.video_frames - 1) % 4:
        p.error("video-frames must be 4n+1")
    try:
        temporal_chunk_ranges(args.frames, args.chunk_size, args.fixed_first_frame)
    except ValueError as error:
        p.error(str(error))
    if args.fixed_first_frame and args.frames < 2:
        p.error("Fixed-first-frame training needs at least one generated latent frame")
    if not args.fixed_first_frame and args.frames % args.chunk_size:
        p.error("Legacy training requires latent frames divisible by chunk-size")
    spatial_multiple = spec["spatial_downscale"] * 2
    if (
        min(args.height, args.width) < spatial_multiple
        or args.height % spatial_multiple
        or args.width % spatial_multiple
    ):
        p.error(f"height/width must be positive multiples of {spatial_multiple}")
    if (
        min(args.steps, args.iterations, args.save_every, args.student_update_freq, args.ode_substeps, args.ode_samples)
        < 1
    ):
        p.error("steps, iterations, frequencies and ODE counts must be positive")
    if args.stage == "dmd" and args.student_update_freq < 2:
        p.error("DMD must update fake-score: student-update-freq >= 2")
    if args.streaming_student_backward and args.stage != "dmd":
        p.error("--streaming-student-backward is only supported for DMD")
    if not 1 <= args.layers <= spec["num_layers"] or (
        args.layers != spec["num_layers"] and not (args.synthetic_data or args.smoke_test)
    ):
        p.error("Truncated models require --synthetic-data or --smoke-test")
    if sum((bool(args.data), bool(args.raw_lmdb), args.synthetic_data)) != 1:
        p.error("Choose exactly one of --data, --raw-lmdb or --synthetic-data")
    if args.raw_lmdb and (not args.text_cache or args.stage not in ("sft", "dmd")):
        p.error("--raw-lmdb requires --text-cache and supports SFT/DMD; KD needs ODE trajectories")
    if args.text_cache and not args.raw_lmdb:
        p.error("--text-cache requires --raw-lmdb")
    if args.allow_cross_control_ode and (args.stage != "kd" or not args.data):
        p.error("--allow-cross-control-ode requires KD --data with compact ODE trajectories")
    if args.num_workers < 0:
        p.error("--num-workers must be non-negative")
    if args.resume and args.student_init:
        p.error("--resume and --student-init are mutually exclusive")
    if args.stage in ("dmd", "ode") and not (args.student_init or args.resume or args.synthetic_data):
        p.error("Real DMD/ODE requires --student-init from SFT/KD (or --resume)")
    if args.stage == "ode" and args.resume:
        p.error("ODE generation uses --student-init, not training resume")
    try:
        validate_shift(args.schedule_shift)
        if args.t_list is None:
            # SFT does not consume this grid; preserve its existing resume contract.
            args.t_list = student_sigmas(args.steps, 1.0 if args.stage == "sft" else args.schedule_shift)
    except ValueError as error:
        p.error(str(error))
    if (
        len(args.t_list) != args.steps + 1
        or not all(math.isfinite(t) for t in args.t_list)
        or args.t_list[-1] != 0
        or args.t_list[0] > 0.999
        or any(a <= b for a, b in zip(args.t_list, args.t_list[1:]))
    ):
        p.error("t-list must contain steps+1 strictly decreasing RF sigmas in [0,0.999], ending at zero")
    if not math.isclose(args.t_list[0], 0.999, rel_tol=0, abs_tol=1e-9):
        p.error("t-list must start at 0.999 to match FastGen Self-Forcing's initial noise scale")
    if args.stage == "ode":
        try:
            euler_sigmas(args.t_list, args.ode_substeps, args.schedule_shift)
        except ValueError as error:
            p.error(str(error))
    return args


def architecture(model_type, layers):
    architecture = dict(get_model_spec(model_type)["architecture"])
    architecture["num_layers"] = layers
    return architecture


def _optimizer_config(lr, betas):
    return L(get_optimizer)(
        model=None, optim_type="adamw", lr=lr, weight_decay=0.01,
        betas=tuple(betas), eps=1e-8, fused=False, foreach=False,
    )


def _scheduler_config():
    return L(LambdaLinearScheduler)(
        warm_up_steps=[0], cycle_lengths=[10_000_000_000],
        f_start=[1.0], f_max=[1.0], f_min=[1.0],
    )


def make_model(args):
    model_cls = {
        "sft": CausalSFTModel,
        "ode": CausalSFTModel,
        "kd": CausalKDModel,
        "dmd": SelfForcingModel,
    }[args.stage]
    spec = get_model_spec(args.model_type)
    common = dict(
        model_id_or_local_path=spec["model_id"],
        load_pretrained=False,
        transformer_config=architecture(args.model_type, args.layers),
        enable_logvar_linear=False,
        use_fsdp_checkpoint=True,
        disable_grad_ckpt=False,
        use_wan_official_sinusoidal=True,
    )
    net = L(CausalWanI2V)(
        **common, chunk_size=args.chunk_size, total_num_frames=args.frames, delete_cache_on_clear=True,
        fixed_first_frame=args.fixed_first_frame,
    )
    sample_t_cfg = SimpleNamespace(
        min_t=0.001, max_t=0.999, time_dist_type="uniform", shift=5.0,
        train_p_mean=-1.1, train_p_std=2.0, t_list=args.t_list, log_t_df=0.01,
    )
    cfg = SimpleNamespace(
        net=net,
        teacher=L(WanI2V)(**common) if args.stage == "dmd" else None,
        device="cuda",
        input_shape=[spec["latent_channels"], args.frames, args.height // spec["spatial_downscale"],
                     args.width // spec["spatial_downscale"]],
        precision="bfloat16", precision_fsdp=args.storage_dtype,
        precision_amp=None, precision_amp_infer=None, precision_amp_enc=None,
        enable_preprocessors=False, load_student_weights=False, use_ema=False,
        fsdp_meta_init=True, fsdp_meta_init_all_ranks=True, add_teacher_to_fsdp_dict=True,
        student_sample_steps=args.steps, student_sample_type="sde",
        guidance_scale=args.guidance_scale, skip_layers=None, sample_t_cfg=sample_t_cfg,
        net_optimizer=_optimizer_config(args.lr, args.adam_betas),
        net_scheduler=_scheduler_config(),
        pretrained_model_path="", pretrained_student_net_path="",
        grad_scaler_enabled=False, grad_scaler_init_scale=65536.0,
        grad_scaler_growth_interval=2000, torch_compile_mode=None,
        cond_dropout_prob=None, cond_keys_no_dropout=[], context_noise=0.0,
    )
    if args.stage == "dmd":
        cfg.sample_t_cfg.time_dist_type = "shifted" if args.schedule_shift > 1 else "uniform"
        cfg.sample_t_cfg.shift = args.schedule_shift
    if args.stage in ("sft", "ode"):
        cfg.cond_dropout_prob = 0.1
        cfg.cond_keys_no_dropout = ["first_frame_cond", "control_hidden_states"]
        if spec["use_image_encoder"]:
            cfg.cond_keys_no_dropout.append("encoder_hidden_states_image")
    if args.stage == "dmd":
        cfg.reset_peak_memory_in_rollout = False  # trainer measures the entire update
        cfg.student_update_freq = args.student_update_freq
        cfg.start_gradient_frame = 0
        cfg.last_step_only = False
        cfg.same_step_across_blocks = True
        cfg.enable_gradient_in_rollout = True
        cfg.gan_loss_weight_gen = cfg.gan_r1_reg_weight = 0.0
        cfg.gan_use_same_t_noise = False
        cfg.gan_r1_reg_alpha = 0.1
        cfg.fake_score_pred_type = None
        cfg.fake_score_sample_t_cfg = None
        cfg.discriminator = None
        cfg.fake_score_optimizer = _optimizer_config(args.fake_score_lr, args.adam_betas)
        cfg.fake_score_scheduler = _scheduler_config()
        cfg.discriminator_optimizer = _optimizer_config(args.fake_score_lr, args.adam_betas)
        cfg.discriminator_scheduler = _scheduler_config()
    return model_cls(cfg)


def log(event, **fields):
    if dist.get_rank() == 0:
        print("CONTROL_TRAIN " + json.dumps({"event": event, "rank": 0, **fields}), flush=True)


def memory():
    return {
        "allocated_GiB": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_GiB": round(torch.cuda.memory_reserved() / 2**30, 3),
        "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }


def materialize(network, mesh, args, role, checkpoint):
    log("materialize_begin", role=role, **memory())
    if any(p.device.type != "meta" for p in network.parameters()):
        raise RuntimeError(f"{role} was not built entirely on meta")
    storage_dtype = torch.bfloat16 if role == "teacher" else getattr(torch, args.storage_dtype)
    network.to(dtype=storage_dtype)
    offload = role in args.cpu_offload_roles
    network.fully_shard(
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
        ),
        offload_policy=CPUOffloadPolicy(pin_memory=False) if offload else None,
        reshard_after_forward=True,
    )
    network.to_empty(device="cpu" if offload else torch.cuda.current_device())
    network.reset_nonpersistent_state()
    network.set_noise_schedule()
    if hasattr(network.noise_scheduler, "_sigmas"):
        network.noise_scheduler._sigmas = network.noise_scheduler._sigmas.cuda()
    summary = stream_wan_official_local_state(
        network.transformer,
        args.weights,
        checkpoint=checkpoint,
        max_blocks=args.layers if args.layers != get_model_spec(args.model_type)["num_layers"] else None,
    )
    log("loaded", role=role, summary=summary, **memory())


def _materialize_unsharded_student(network):
    """Materialize one complete BF16 student per inference worker."""
    if any(parameter.device.type != "meta" for parameter in network.parameters()):
        raise RuntimeError("Checkpoint student must be constructed entirely on meta")
    network.to(dtype=torch.bfloat16)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_nonpersistent_state()
    network.set_noise_schedule()
    if hasattr(network.noise_scheduler, "_sigmas"):
        network.noise_scheduler._sigmas = network.noise_scheduler._sigmas.cuda()


def _factory_args(config, *, stage):
    settings = config["distillation"]
    training = config["training"]
    steps = int(settings.get("steps", 4))
    shift = float(settings.get("schedule_shift", 5.0))
    t_list = settings.get("t_list") or student_sigmas(steps, 1.0 if stage == "sft" else shift)
    video_frames = int(training.get("num_frames", 81))
    spec = get_model_spec(config["model_type"])
    return SimpleNamespace(
        stage=stage,
        model_type=config["model_type"],
        layers=spec["num_layers"],
        chunk_size=int(settings.get("chunk_size", 4)),
        frames=(video_frames - 1) // spec["temporal_downscale"] + 1,
        fixed_first_frame=bool(settings.get("fixed_first_frame", True)),
        steps=steps,
        height=int(training["height"]),
        width=int(training["width"]),
        storage_dtype=settings.get("storage_dtype", "float32"),
        guidance_scale=float(settings.get("guidance_scale", 1.0)),
        schedule_shift=shift,
        t_list=list(t_list),
        lr=float(settings.get("lr", 2e-6)),
        adam_betas=list(settings.get("adam_betas") or ([0.0, 0.999] if stage == "dmd" else [0.9, 0.999])),
        student_update_freq=int(settings.get("student_update_freq", 6)),
        fake_score_lr=float(settings.get("fake_score_lr", 4e-7)),
        cpu_offload_roles=[],
        sharding_strategy=settings.get("sharding_strategy", "full"),
    )


def _ensure_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("Distillation model loading requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Launch distillation model loading with torchrun")
        dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=2))


def validate_student_contract(config, contract, control_digest, args):
    """Validate and apply the immutable causal schedule from a DCP checkpoint."""
    settings = config["distillation"]
    if contract.get("model_type") != config["model_type"]:
        raise ValueError("Student checkpoint model_type differs from the inference recipe")
    if control_representation(contract) != config["representation"]:
        raise ValueError("Student checkpoint control representation differs from the inference recipe")
    if contract.get("control_sha256") != control_digest:
        raise ValueError("Student checkpoint was trained with a different control checkpoint")
    if contract.get("chunk_size") != args.chunk_size or bool(
        contract.get("fixed_first_frame", False)
    ) != args.fixed_first_frame:
        raise ValueError("Student checkpoint causal chunk boundaries differ from the inference recipe")
    if contract.get("stage") not in ("kd", "dmd"):
        return args
    checkpoint_times = list(contract.get("t_list") or [])
    configured_times = settings.get("t_list")
    if configured_times is not None and list(configured_times) != checkpoint_times:
        raise ValueError("Configured t_list differs from the distilled student checkpoint")
    if len(checkpoint_times) < 2:
        raise ValueError("Distilled student checkpoint has no valid timestep schedule")
    if int(settings.get("steps", len(checkpoint_times) - 1)) != len(checkpoint_times) - 1:
        raise ValueError("Configured step count differs from the distilled student checkpoint")
    checkpoint_shift = contract.get("schedule_shift")
    if checkpoint_shift is not None and not math.isclose(
        float(settings.get("schedule_shift", checkpoint_shift)), float(checkpoint_shift)
    ):
        raise ValueError("Configured schedule_shift differs from the distilled student checkpoint")
    args.t_list = checkpoint_times
    args.steps = len(checkpoint_times) - 1
    return args


def build_inference_model(config, config_path=None, checkpoint=None):
    """Load a completed causal student for ODE generation or inference.

    The DCP student already contains the merged backbone and adapter-conditioned
    student weights. Only the standalone frozen control encoder is read from
    the SFT checkpoint; the LoRA is not merged a second time.
    """
    del config_path
    _ensure_distributed()
    settings = config["distillation"]
    checkpoint = Path(checkpoint or settings.get("student_init") or "").expanduser().resolve()
    if not (checkpoint / "complete.json").is_file():
        raise ValueError(f"Completed student checkpoint required: {checkpoint}")
    metadata = json.loads((checkpoint / "complete.json").read_text())
    contract = metadata.get("contract", metadata)
    portable = (checkpoint / "student.safetensors").is_file()
    if portable:
        if metadata.get("format") != "handsonworld_causal_student_v1":
            raise ValueError("Unsupported portable student bundle format")
        for name, field in (
            ("student.safetensors", "student_sha256"),
            ("adapter.safetensors", "adapter_sha256"),
        ):
            expected = metadata.get(field)
            path = checkpoint / name
            if not path.is_file() or not expected or file_digest(path) != expected:
                raise ValueError(f"Portable bundle hash mismatch: {name}")
        control_digest = metadata.get("source_control_sha256") or contract.get("source_control_sha256")
        if not control_digest:
            raise ValueError("Portable bundle is missing source_control_sha256")
        control_path = checkpoint / "adapter.safetensors"
    else:
        control_path = Path(settings["teacher_checkpoint"])
        control_digest = file_digest(control_path)
    args = _factory_args(config, stage="ode")
    validate_student_contract(config, contract, control_digest, args)
    model = make_model(args)
    _materialize_unsharded_student(model.net)
    if portable:
        load_safetensors_student(model.net, checkpoint / "student.safetensors")
    else:
        load_unsharded_student(model.net, checkpoint)
    model.net.eval().requires_grad_(False)

    control_checkpoint = DiffSynthControlCheckpoint(control_path, allow_adapter_only=portable)
    adapter_holder = [None]
    from handsonworld.distillation.raw_condition import RawConditioner

    conditioner = RawConditioner(
        config["backend_root"], args.height, args.width,
        representation=config["representation"], device="cuda", dtype=torch.bfloat16,
    )

    def prepare_data(data):
        return prepare_batch(data, control_checkpoint, adapter_holder, args, conditioner, False)

    def sample_data(data, *, steps=None, t_list=None, schedule_shift=None):
        prepared = prepare_data(data)
        _, condition, _ = model._prepare_training_data(prepared)
        count = int(args.steps if steps is None else steps)
        if schedule_shift is not None and not math.isclose(float(schedule_shift), args.schedule_shift):
            raise ValueError("Inference schedule_shift differs from the loaded student checkpoint")
        boundaries = args.t_list if t_list is None else t_list
        if len(boundaries) != count + 1:
            raise ValueError("Inference step count differs from the loaded student timestep schedule")
        noise = torch.randn_like(prepared["real"])
        return CausVidModel.generator_fn(
            net=model.net,
            noise=noise,
            condition=condition,
            student_sample_steps=count,
            student_sample_type="sde",
            t_list=boundaries,
            precision_amp=None,
            context_noise=0.0,
        )

    model.prepare_inference_data = prepare_data
    model.sample_data = sample_data
    model.requires_collective_forward = False
    return model


def build_model_for_stage(config, config_path=None):
    """Build the causal teacher used by the public ODE worker."""
    return build_inference_model(
        config,
        config_path,
        checkpoint=config["distillation"].get("student_init"),
    )


def synthetic(args, iteration):
    # Local generator avoids perturbing the training RNG and is stable across resume.
    gen = torch.Generator().manual_seed(args.seed + iteration * dist.get_world_size() + dist.get_rank())
    spec = get_model_spec(args.model_type)
    shape = (
        1, spec["latent_channels"], args.frames,
        args.height // spec["spatial_downscale"], args.width // spec["spatial_downscale"],
    )

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.bfloat16)

    data = dict(
        real=rand(*shape),
        condition=rand(1, 512, spec["text_dim"]),
        neg_condition=rand(1, 512, spec["text_dim"]),
        first_frame_cond=rand(*shape),
        packed_control=rand(1, 48, args.frames, args.height, args.width),
        valid_mask=torch.ones(1),
    )
    if spec["use_image_encoder"]:
        data["encoder_hidden_states_image"] = rand(
            1, spec["image_context_tokens"], spec["image_context_dim"]
        )
    if args.stage == "kd":
        noise = rand(*shape)
        data["path"] = torch.stack([(1 - t) * data["real"] + t * noise for t in args.t_list[:-1]], dim=1)
        data["path_timesteps"] = torch.tensor(args.t_list[:-1]).unsqueeze(0)
    return data


def prepare_batch(data, checkpoint, adapter_holder, args, raw_conditioner=None, adapter_cpu_offload=False):
    if raw_conditioner is not None:
        data["packed_control"], data["valid_mask"] = raw_conditioner(data)
        for key in (
            "raw_cam_pose", "raw_hand_pose", "raw_camera_intrinsics",
            "raw_original_height", "raw_original_width",
        ):
            data.pop(key, None)
    if "control_hidden_states" not in data:
        # A frozen encoder outside model/optimizer trees cannot accidentally be
        # unfrozen by DMD's fake_score.requires_grad_(True).
        if adapter_holder[0] is None:
            spec = get_model_spec(args.model_type)
            adapter_holder[0] = checkpoint.load_adapter(
                device="cpu" if adapter_cpu_offload else "cuda",
                dtype=torch.bfloat16,
                in_dim=spec["control_channels"],
                out_dim=spec["dim"],
                kernel_size=spec["patch_size"][1:],
                stride=spec["patch_size"][1:],
                downscale_factor=spec["spatial_downscale"],
            )
        adapter = adapter_holder[0]
        if adapter_cpu_offload:
            adapter.cuda()
        data["control_hidden_states"] = encode_control(adapter, data.pop("packed_control"), data.pop("valid_mask"))
        if adapter_cpu_offload:
            adapter.cpu()
    return {
        key: value.to(device="cuda", dtype=torch.bfloat16)
        if isinstance(value, torch.Tensor) and key != "path_timesteps"
        else value
        for key, value in data.items()
    }


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_contract(args):
    digest = [file_digest(args.control_checkpoint) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(digest, src=0)
    representation = args.control_representation
    validate_control_metadata(args.control_checkpoint, representation, digest[0], args.model_type)
    keys = (
        "stage",
        "model_type",
        "frames",
        "height",
        "width",
        "layers",
        "chunk_size",
        "t_list",
        "storage_dtype",
        "lora_scale",
        "guidance_scale",
        "student_update_freq",
        "seed",
        "lr",
        "max_grad_norm",
        "synthetic_data",
    )
    contract = {key: getattr(args, key) for key in keys}
    if args.stage != "sft":
        contract["schedule_shift"] = args.schedule_shift
    if args.stage == "ode":
        contract["ode_solver"] = "euler"
        contract["ode_sigmas"] = euler_sigmas(args.t_list, args.ode_substeps, args.schedule_shift)
    if args.stage == "dmd":
        contract["score_time_distribution"] = "shifted" if args.schedule_shift > 1 else "uniform"
        contract["vsd_loss_precision"] = "float32"
        contract["fake_score_lr"] = args.fake_score_lr
        contract["adam_betas"] = list(args.adam_betas)
    if args.fixed_first_frame:
        contract["fixed_first_frame"] = True
    if args.streaming_student_backward:
        contract["streaming_student_backward"] = True
    contract.update(
        control_sha256=digest[0],
        weights=str(args.weights.resolve()),
        data=str(args.data.resolve()) if args.data else None,
    )
    # Keep legacy reflection contracts unchanged so existing resumes work.
    if representation != "reflection_plucker":
        contract.update(control_representation=representation,
                        control_checkpoint=str(args.control_checkpoint.resolve()))
    if args.data:
        manifest_path = args.data if args.data.is_file() else args.data / "index.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            source_control = validate_ode_control(
                manifest, contract, allow_cross_control=args.allow_cross_control_ode,
            )
            if source_control is not None:
                contract["ode_source_control"] = source_control
            contract["data_manifest_sha256"] = file_digest(manifest_path)
            if args.stage == "kd":
                trajectory_contract = manifest.get("provenance", {})
                if bool(trajectory_contract.get("fixed_first_frame", False)) != args.fixed_first_frame:
                    raise ValueError("KD trajectories use a different fixed-first-frame policy; regenerate ODE data")
                if args.fixed_first_frame and trajectory_contract.get("chunk_size") != args.chunk_size:
                    raise ValueError("KD trajectory chunk size differs; regenerate ODE data with matching boundaries")
    if args.raw_lmdb:
        contract.update(
            raw_lmdb=[str(path.resolve()) for path in args.raw_lmdb],
            text_cache=str(args.text_cache.resolve()),
            text_cache_manifest_sha256=file_digest(args.text_cache / "index.json"),
        )
    return contract


@torch.no_grad()
def generate_trajectory(model, data, args, *, include_conditions=True, reuse_kv_cache=False):
    """Compatibility wrapper around the canonical public ODE generator."""
    from handsonworld.distillation.ode import generate_trajectory as generate_public_trajectory

    result = generate_public_trajectory(
        model, data, args, reuse_kv_cache=reuse_kv_cache,
    )
    if include_conditions:
        conditions = {
            key: value[0].detach().cpu().contiguous()
            for key, value in data.items()
            if isinstance(value, torch.Tensor) and key not in ("real", "path", "path_timesteps")
        }
        conditions.update(result)
        return conditions
    return result


@record
@record_local_errors
def main():
    args = parse_args()
    if args.preallocate_checkpoints is None:
        args.preallocate_checkpoints = args.stage == "kd"
    if args.stage == "ode" and args.preallocate_checkpoints:
        raise ValueError("Use the ODE output reservation instead of checkpoint preallocation")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=2))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed + dist.get_rank())
    torch.cuda.manual_seed_all(args.seed + dist.get_rank())
    output_lock = None
    try:
        from handsonworld.distillation.checkpoint_reservation import acquire_output_lock, reserve_checkpoints

        output_lock = acquire_output_lock(args.output)
        if args.stage in ("sft", "kd"):
            from diffsynth.models.wan_video_dit import FLEX_ATTENTION_AVAILABLE

            if not FLEX_ATTENTION_AVAILABLE:
                raise RuntimeError("Causal SFT/KD on CUDA requires PyTorch FlexAttention")
        mesh = build_training_mesh(args.sharding_strategy)
        shard_group = mesh.get_group("shard")
        contract = training_contract(args)
        # Keep existing single-node/full-shard checkpoints compatible. Exact
        # HSDP resumes must retain the replica/shard layout as well as world size.
        if mesh.ndim == 2:
            contract["fsdp_mesh_shape"] = list(mesh.shape)
        args.output.mkdir(parents=True, exist_ok=True)
        if dist.get_rank() == 0:
            run_config = dict(vars(args))
            if args.stage == "dmd":
                run_config["vsd_loss_precision"] = contract["vsd_loss_precision"]
            (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")
        log(
            "start",
            contract=contract,
            world_size=dist.get_world_size(),
            hostname=socket.gethostname(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            torch_reported_nccl=torch.cuda.nccl.version(),
            gpu=torch.cuda.get_device_name(),
            compile_mode=os.environ.get("TORCH_COMPILE_MODE"),
            allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            fixed_first_frame=args.fixed_first_frame,
            temporal_chunks=temporal_chunk_ranges(args.frames, args.chunk_size, args.fixed_first_frame),
            sharding_strategy=args.sharding_strategy,
            mesh_shape=list(mesh.shape),
            mesh_dim_names=mesh.mesh_dim_names,
            shard_ranks=dist.get_process_group_ranks(shard_group),
        )
        loader = sampler = None
        steps_per_epoch = None
        compact_ode = False
        if args.data or args.raw_lmdb:
            if args.raw_lmdb:
                raise ValueError(
                    "Direct LMDB/text-cache training is a legacy input. Run the public prepare stage and pass --data."
                )
            manifest_path = args.data if args.data.is_file() else args.data / "index.json"
            compact_ode = args.stage == "kd"
            if compact_ode:
                from handsonworld.data.trajectories import ODEDataset

                dataset = ODEDataset(
                    manifest_path,
                    args.model_type,
                    expected_path_timesteps=args.t_list[:-1],
                    expected_control_representation=args.control_representation,
                )
            else:
                from handsonworld.data.cache import CacheDataset

                dataset = CacheDataset(
                    args.data,
                    args.model_type,
                    expected_control_representation=args.control_representation,
                )
            sampler = ResumeDistributedSampler(dataset, shuffle=args.stage != "ode", seed=args.seed, drop_last=False)
            loader = DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=False,
                generator=torch.Generator().manual_seed(args.seed),
            )
            # Keep the full epoch size: a resumed sampler exposes only its tail.
            steps_per_epoch = len(loader)
            log(
                "dataset",
                samples=len(dataset),
                batches_per_rank=len(loader),
            format="ode" if compact_ode else "cache",
            )
        checkpoint = DiffSynthControlCheckpoint(args.control_checkpoint, lora_scale=args.lora_scale)
        model = make_model(args)
        roles = ["net", "teacher", "fake_score"] if args.stage == "dmd" else ["net"]
        for role in roles:
            materialize(getattr(model, role), mesh, args, role, checkpoint)
        if args.student_init:
            previous = json.loads((args.student_init / "complete.json").read_text())["contract"]
            if control_representation(previous) != control_representation(contract):
                raise ValueError("Student initialization mismatch: control_representation")
            for key in ("model_type", "layers", "control_sha256", "lora_scale", "weights"):
                if previous[key] != contract[key]:
                    raise ValueError(f"Student initialization mismatch: {key}")
            if args.stage != "sft" and (
                previous["chunk_size"] != args.chunk_size
                or bool(previous.get("fixed_first_frame", False)) != args.fixed_first_frame
            ):
                raise ValueError("Student initialization uses different causal boundaries; adapt it with SFT first")
            if previous["stage"] in ("kd", "dmd") and previous["t_list"] != contract["t_list"]:
                raise ValueError("Few-step student checkpoint schedule differs from requested t-list")
            load_student(model.net, args.student_init)
            log("student_initialized", path=str(args.student_init))
        model.net.train()
        if args.stage != "ode":
            model.init_optimizers()
        optimizers = {"net": model.net_optimizer} if args.stage != "ode" else {}
        schedulers = {"net": model.net_lr_scheduler} if args.stage != "ode" else {}
        if args.stage == "dmd":
            optimizers["fake_score"] = model.fake_score_optimizer
            schedulers["fake_score"] = model.fake_score_lr_scheduler
        iteration = resume_checkpoint(args.resume, model, optimizers, schedulers, contract) if args.resume else 0
        scheduled_saves = set(checkpoint_steps(iteration, args.iterations, args.save_every, args.save_steps))
        if dist.get_rank() == 0:
            print(f"CHECKPOINT save steps={sorted(scheduled_saves)}; training continues through saves", flush=True)
            if args.stage == "dmd":
                print(f"DMD optimizer recipe: student_lr={args.lr} fake_score_lr={args.fake_score_lr} "
                      f"betas={args.adam_betas} fake:student={args.student_update_freq - 1}:1 "
                      f"storage={args.storage_dtype} compute=bfloat16", flush=True)
        if args.preallocate_checkpoints:
            reserve_checkpoints(args.output, model, optimizers, iteration, args.iterations, args.save_every, contract,
                                save_optimizer=args.save_optimizer, save_steps=args.save_steps)
        data_iterator = None
        adapter_holder = [None]
        raw_conditioner = None
        if loader is not None:
            from handsonworld.distillation.raw_condition import RawConditioner

            raw_conditioner = RawConditioner(
                args.diffsynth_root, args.height, args.width,
                representation=args.control_representation, device="cuda", dtype=torch.bfloat16,
            )
        total = (
            (args.ode_samples + dist.get_world_size() - 1) // dist.get_world_size()
            if args.stage == "ode"
            else args.iterations
        )
        ode_files = []
        progress = StepProgress(
            args.stage, total, initial=iteration,
            steps_per_epoch=steps_per_epoch if args.stage != "ode" else None,
        ) if args.progress and dist.get_rank() == 0 else None
        if progress is not None:
            print(f"PROGRESS {args.stage.upper()} starting at {iteration}/{total}; "
                  f"steps_per_epoch={steps_per_epoch if loader is not None else 'synthetic'}; "
                  f"global_batch={dist.get_world_size()}; first steps may include compilation", flush=True)
        while iteration < total:
            step_started = time.monotonic()
            log("batch_begin", iteration=iteration + 1, **memory())
            if loader is not None:
                offset = iteration % steps_per_epoch
                if data_iterator is None or offset == 0:
                    epoch = iteration // steps_per_epoch
                    sampler.set_epoch(epoch, start_index=offset)
                    log("data_position", epoch=epoch, sample_offset=offset,
                        steps_per_epoch=steps_per_epoch, iteration=iteration + 1)
                    data_iterator = iter(loader)
                data = next(data_iterator)
            else:
                data = synthetic(args, iteration)
            log("batch_loaded", iteration=iteration + 1, seconds=round(time.monotonic() - step_started, 3))
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            data = prepare_batch(
                data, checkpoint, adapter_holder, args, raw_conditioner, args.adapter_cpu_offload
            )
            if args.stage == "ode":
                # All ranks must execute equal forward counts for FSDP, even
                # if the final global batch contains padding samples.
                result = generate_trajectory(model, data, args)
                index = iteration * dist.get_world_size() + dist.get_rank()
                if index < args.ode_samples:
                    name = f"trajectory-{index:08d}.safetensors"
                    if (args.output / name).exists():
                        raise FileExistsError(args.output / name)
                    save_file(result, str(args.output / name))
                    ode_files.append(name)
                log("trajectory", index=index, seconds=time.monotonic() - started, **memory())
                del result, data
                iteration += 1
                if args.progress:
                    torch.cuda.synchronize()
                    step_seconds = global_step_seconds(step_started, torch.cuda.current_device())
                    if progress is not None:
                        print(progress.update(iteration, step_seconds, phase="trajectory"), flush=True)
                continue
            log("forward_begin", iteration=iteration + 1, **memory())
            streaming = args.streaming_student_backward and iteration % args.student_update_freq == 0
            if streaming:
                from handsonworld.distillation.streaming import streaming_student_backward

                losses, outputs = streaming_student_backward(
                    model,
                    data,
                    iteration,
                    progress=lambda event, **fields: log(event, iteration=iteration + 1, **fields, **memory()),
                )
            else:
                losses, outputs = model.single_train_step(data, iteration)
            loss = losses["total_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            if not streaming:
                log("backward_begin", iteration=iteration + 1, **memory())
                loss.backward()
            log("optimizer_begin", iteration=iteration + 1, **memory())
            active_optimizers = model.get_optimizers(iteration)
            norm = clip_grad(active_optimizers, args.max_grad_norm, shard_group=shard_group)
            model.optimizers_schedulers_step(iteration)
            phase = "student" if args.stage != "dmd" or iteration % args.student_update_freq == 0 else "fake_score"
            torch.cuda.synchronize()
            optimizer_steps = [max(float(state["step"]) for state in opt.state.values()) for opt in active_optimizers]
            loss_value = loss.item()
            log(
                "update",
                iteration=iteration + 1,
                phase=phase,
                loss=loss_value,
                grad_norm=norm,
                optimizer_steps=optimizer_steps,
                seconds=round(time.monotonic() - started, 3),
                **memory(),
            )
            del loss, losses, outputs, data
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            iteration += 1
            if args.progress:
                step_seconds = global_step_seconds(step_started, torch.cuda.current_device())
                if progress is not None:
                    print(progress.update(iteration, step_seconds, loss=loss_value, phase=phase), flush=True)
            if iteration in scheduled_saves:
                destination = args.output / f"step-{iteration:07d}"
                log("save_begin", path=str(destination), **memory())
                save_started = time.monotonic()
                if progress is not None:
                    print(f"PROGRESS {args.stage.upper()} saving checkpoint {destination}", flush=True)
                save_checkpoint(destination, model, optimizers, schedulers, iteration, contract,
                                preallocated=args.preallocate_checkpoints, save_optimizer=args.save_optimizer)
                log("saved", path=str(destination), **memory())
                if progress is not None:
                    print(f"PROGRESS {args.stage.upper()} checkpoint saved in {time.monotonic() - save_started:.1f}s", flush=True)
        if args.stage == "ode":
            all_files = [None] * dist.get_world_size()
            dist.all_gather_object(all_files, ode_files)
            if dist.get_rank() == 0:
                (args.output / "index.json").write_text(
                    json.dumps(
                        {
                            "samples": sorted(sum(all_files, [])),
                            "path_timesteps": args.t_list[:-1],
                            "provenance": contract,
                            "teacher_checkpoint": str(args.student_init),
                            "ode_substeps": args.ode_substeps,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        log("complete", stage=args.stage, iterations=iteration)
    finally:
        if output_lock is not None:
            output_lock.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
