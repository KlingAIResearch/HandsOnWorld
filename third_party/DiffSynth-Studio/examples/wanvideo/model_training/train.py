# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import torch, os, argparse, accelerate, warnings, re
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import (
    LoadVideo, LoadAudio, LoadNpyFile, ImageCropAndResize, ToAbsolutePath,
    resolve_original_video_size,
)
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_TOKENIZER_PATH = os.path.join(
    REPO_ROOT,
    "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
)


def nvtx_push(name):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def find_latest_checkpoint(output_path):
    if not os.path.isdir(output_path):
        raise FileNotFoundError("Checkpoint directory does not exist: %s" % output_path)

    checkpoints = [
        os.path.join(output_path, filename)
        for filename in os.listdir(output_path)
        if filename.endswith(".safetensors")
    ]
    if not checkpoints:
        raise FileNotFoundError("No .safetensors checkpoint found in %s" % output_path)

    def checkpoint_key(path):
        filename = os.path.basename(path)
        match = re.match(r"^(epoch|step)-(\d+)\.safetensors$", filename)
        if match is None:
            return (0, os.path.getmtime(path), -1)
        kind, index = match.groups()
        return (1, 1 if kind == "step" else 0, int(index), os.path.getmtime(path))

    return max(checkpoints, key=checkpoint_key)


def parse_checkpoint_progress(checkpoint_path):
    if checkpoint_path is None:
        return 0, 0
    match = re.match(r"^(epoch|step)-(\d+)\.safetensors$", os.path.basename(checkpoint_path))
    if match is None:
        return 0, 0
    kind, index = match.groups()
    index = int(index)
    if kind == "epoch":
        return index + 1, 0
    return 0, index


PIPE_MODULE_NAMES = {
    "dit",
    "dit2",
    "pose_controller",
    "hand_motion_controller",
    "pose_video_controller",
    "pose_video_controller2",
    "motion_controller",
    "vace",
    "vace2",
    "animate_adapter",
    "vap",
}


def normalize_checkpoint_state_dict(state_dict):
    if any(key.startswith("pipe.") for key in state_dict):
        return {
            key[len("pipe."):] if key.startswith("pipe.") else key: value
            for key, value in state_dict.items()
        }
    return state_dict


def strip_module_prefix(state_dict, module_name):
    prefix = module_name + "."
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def is_lora_checkpoint(state_dict):
    lora_patterns = (
        ".lora_A.",
        ".lora_B.",
        ".lora_down.",
        ".lora_up.",
        ".lora_A.default.",
        ".lora_B.default.",
    )
    return any(any(pattern in key for pattern in lora_patterns) for key in state_dict)


def checkpoint_modules(state_dict):
    modules = set()
    for key in state_dict:
        module_name = key.split(".", 1)[0]
        if module_name in PIPE_MODULE_NAMES:
            modules.add(module_name)
    return modules


def expected_checkpoint_modules(trainable_models=None, lora_base_model=None):
    modules = set()
    if trainable_models is not None:
        modules.update(
            module.split(".", 1)[0]
            for module in trainable_models.split(",")
            if module
        )
    if lora_base_model is not None:
        modules.add(lora_base_model.split(".", 1)[0])
    return modules


def validate_checkpoint_modules(modules, expected_modules, resume):
    if not expected_modules or not modules:
        return
    unexpected_modules = modules - expected_modules
    if unexpected_modules:
        raise ValueError(
            "Resume checkpoint %s contains modules %s, but this run only expects %s"
            % (resume, sorted(unexpected_modules), sorted(expected_modules))
        )
    missing_modules = expected_modules - modules
    if missing_modules:
        print(
            "Warning, resume checkpoint %s does not contain expected modules: %s"
            % (resume, sorted(missing_modules))
        )


def load_resume_checkpoint(pipe, resume, trainable_models=None, lora_base_model=None):
    state_dict = normalize_checkpoint_state_dict(load_state_dict(resume))
    modules = checkpoint_modules(state_dict)
    expected_modules = expected_checkpoint_modules(trainable_models, lora_base_model)
    is_lora = is_lora_checkpoint(state_dict)
    validate_checkpoint_modules(modules, expected_modules, resume)

    if is_lora and lora_base_model is None:
        raise ValueError(
            "Resume checkpoint contains LoRA keys, but --lora_base_model is not set. "
            "Set --lora_base_model to inject LoRA adapters before loading this checkpoint."
        )

    if modules == {"dit"}:
        target_state_dict = strip_module_prefix(state_dict, "dit")
        missing_keys, unexpected_keys = pipe.dit.load_state_dict(target_state_dict, strict=False)
        checkpoint_type = "dit-only LoRA" if is_lora else "dit-only"
    elif modules:
        missing_keys, unexpected_keys = pipe.load_state_dict(state_dict, strict=False)
        checkpoint_type = "pipeline LoRA/subset" if is_lora else "pipeline subset"
    else:
        missing_keys, unexpected_keys = pipe.dit.load_state_dict(state_dict, strict=False)
        checkpoint_type = "raw DiT LoRA" if is_lora else "raw DiT"

    print(
        "Resume checkpoint type: %s, modules: %s, expected modules: %s"
        % (
            checkpoint_type,
            sorted(modules) if modules else ["dit"],
            sorted(expected_modules) if expected_modules else "any",
        )
    )
    if unexpected_keys:
        print("Warning, unexpected keys when resuming: ", unexpected_keys)
    return missing_keys, unexpected_keys


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        hand_pose_render_mode="position",
        hand_pose_attr_space="camera",
        resume=None,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = None if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        if resume is not None:
            load_resume_checkpoint(
                self.pipe,
                resume,
                trainable_models=trainable_models,
                lora_base_model=lora_base_model,
            )
            print("Resume from ", resume)
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.hand_pose_render_mode = hand_pose_render_mode
        self.hand_pose_attr_space = hand_pose_attr_space
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared, inputs_posi):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "poses" or extra_input == "pose_video" or extra_input == "cam_pose" or extra_input == "hand_pose" or extra_input == "camera_intrinsics":
                if extra_input in data:
                    inputs_posi[extra_input] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared, inputs_posi
    
    def get_pipeline_inputs(self, data):
        if data["prompt"] != data["prompt"]:
            data["prompt"] = ""
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        original_height, original_width = resolve_original_video_size(
            data["video"], data.get("height"), data.get("width")
        )
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "original_height": original_height,
            "original_width": original_width,
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "hand_pose_render_mode": self.hand_pose_render_mode,
            "hand_pose_attr_space": self.hand_pose_attr_space,
        }
        # print(inputs_shared['height'], inputs_shared['width'])
        inputs_shared, inputs_posi = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared, inputs_posi)
        return inputs_shared, inputs_posi, inputs_nega
    
    def prepare_inputs(self, data):
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            nvtx_push(f"unit_runner/{unit.__class__.__name__}")
            try:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            finally:
                nvtx_pop()
        return inputs

    def compute_loss_from_prepared_inputs(self, inputs):
        nvtx_push("compute_loss_from_prepared_inputs")
        try:
            return self.task_to_loss[self.task](self.pipe, *inputs)
        finally:
            nvtx_pop()

    def forward(self, data, inputs=None, inputs_prepared=False):
        if inputs_prepared:
            return self.compute_loss_from_prepared_inputs(inputs)
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            nvtx_push(f"unit_runner/{unit.__class__.__name__}")
            try:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            finally:
                nvtx_pop()
        return self.task_to_loss[self.task](self.pipe, *inputs)


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Optional explicit audio processor path for S2V; I2V does not load one.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument(
        "--hand_pose_render_mode",
        type=str,
        default="position",
        choices=[
            "position",
            "depth",
            "normal",
            "surface_normal",
            "reflection",
            "reflective_rays",
            "silhouette_wireframe",
            "normal_plucker",
            "reflection_plucker",
            "reflective_rays_plucker",
            "hand_mask",
            "pose9d_mask_broadcast",
            "joint_occ_plucker",
        ],
        help="Hand mesh feature rasterized for hand_pose control adapter.",
    )
    parser.add_argument("--hand_pose_attr_space", type=str, default="camera", choices=["camera", "world"], help="Coordinate space for rasterized hand_pose attributes.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--resume", default=None, type=str, help="Resume from checkpoint(dit)")
    parser.add_argument("--resume_latest", default=False, action="store_true", help="Resume from the latest .safetensors checkpoint in output_path.")
    parser.add_argument("--online_unit_prefetch", default=False, action="store_true", help="Prepare pipeline-unit inputs online between the dataloader and training loop.")
    parser.add_argument("--release_cache_before_backward", default=False, action="store_true", help="Release unused CUDA allocator cache after forward to leave headroom for backward/NCCL; does not offload live tensors.")
    parser.add_argument("--online_unit_prefetch_queue_size", type=int, default=2, help="Maximum number of prepared pipeline-unit inputs to buffer when --online_unit_prefetch is enabled.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    resume_latest = args.resume_latest or args.resume in ("latest", "auto")
    if resume_latest:
        args.resume = find_latest_checkpoint(args.output_path)
        print("Resolved latest checkpoint: ", args.resume)
        args.resume_start_epoch, args.resume_start_step = parse_checkpoint_progress(args.resume)
        print(
            "Resume progress: start_epoch=%d, start_step=%d"
            % (args.resume_start_epoch, args.resume_start_step)
        )
    else:
        args.resume_start_epoch, args.resume_start_step = 0, 0
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        log_with=None
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        random_drop_keys=args.random_drop_keys.split(","),
        random_start_frame=args.random_start_frame,
        sample_num_frames=args.num_frames,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32, # Wan 2.2
            width_division_factor=32,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            resize_mode="resize",
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "camera_intrinsics": ToAbsolutePath(args.dataset_base_path) >> LoadNpyFile(),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        hand_pose_render_mode=args.hand_pose_render_mode,
        hand_pose_attr_space=args.hand_pose_attr_space,
        resume=args.resume,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        start_step=args.resume_start_step,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
