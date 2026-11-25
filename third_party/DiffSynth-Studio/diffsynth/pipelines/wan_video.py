# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import os
import torch, types
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
from tqdm import tqdm
from typing_extensions import Literal
from transformers import Wav2Vec2Processor

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, PipelineUnitRunnerMultiPos

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_pose_controller import WanHandMotionTokenEncoder, WanPoseControllerModel, WanPoseVideoControllerModel
from ..models.wan_video_camera_controller import SimpleAdapter
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.wav2vec import WanS2VAudioEncoder
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel

from .plucker_utils import adjust_intrinsics_for_resize, poses_to_plucker_rays, taco_1080p_intrinsics
from .jointcontrol_utils import JointOccPluckerFeatureBuilder, resize_packed_plucker_to_latent_grid
import copy
from handsonworld.geometry.plucker import surface_directions, plucker_from_direction
from handsonworld.conditioning.control import combine_control_maps


def nvtx_push(name):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.pose_controller: WanPoseControllerModel = None
        self.pose_controller2: WanPoseControllerModel = None
        self.hand_motion_controller: WanHandMotionTokenEncoder = None
        self.hand_motion_controller2: WanHandMotionTokenEncoder = None
        self.pose_video_controller: Union[WanPoseVideoControllerModel, SimpleAdapter] = None
        self.pose_video_controller2: Union[WanPoseVideoControllerModel, SimpleAdapter] = None
        self.motion_controller: WanMotionControllerModel = None
        self.motion_controller2: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None
        self.in_iteration_models = ("dit", "pose_controller", "hand_motion_controller", "pose_video_controller", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "pose_controller2", "hand_motion_controller2", "pose_video_controller2", "motion_controller2", "vace2", "animate_adapter", "vap")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_RebasePoseToClipStart(),
            WanVideoUnit_PoseVideoControl(),
            WanVideoUnit_Pose2Plucker(),
            WanVideoUnit_PosePluckerControlAdapter(),
            WanVideoUnit_HandPosePluckerControlAdapter(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_AnimateVideoSplit(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_VAP(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pose_controller = model_pool.fetch_model("wan_video_pose_controller", index=2)
        if isinstance(pose_controller, list):
            pipe.pose_controller, pipe.pose_controller2 = pose_controller
        else:
            pipe.pose_controller = pose_controller
        hand_motion_controller = model_pool.fetch_model("wan_video_hand_motion_controller", index=2)
        if isinstance(hand_motion_controller, list):
            pipe.hand_motion_controller, pipe.hand_motion_controller2 = hand_motion_controller
        else:
            pipe.hand_motion_controller = hand_motion_controller
        pose_video_controller = model_pool.fetch_model("wan_video_pose_video_controller", index=2)
        if isinstance(pose_video_controller, list):
            pipe.pose_video_controller, pipe.pose_video_controller2 = pose_video_controller
        else:
            pipe.pose_video_controller = pose_video_controller
        motion_controller = model_pool.fetch_model("wan_video_motion_controller", index=2)
        if isinstance(motion_controller, list):
            pipe.motion_controller, pipe.motion_controller2 = motion_controller
        else:
            pipe.motion_controller = motion_controller
        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # VAP
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance, float or list[float]
        cfg_scale: Optional[Union[float, list[float]]] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Pose control
        poses: Optional[np.ndarray] = None,
        cam_pose: Optional[np.ndarray] = None,
        pose_video: Optional[list[Image.Image]] = None,
        hand_pose: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        hand_pose_render_mode: Literal["position", "depth", "normal", "surface_normal", "reflection", "reflective_rays", "silhouette_wireframe", "normal_plucker", "reflection_plucker", "reflective_rays_plucker", "hand_mask", "pose9d_mask_broadcast", "joint_occ_plucker"] = "position",
        hand_pose_attr_space: Literal["camera", "world"] = "camera",
        original_height: Optional[int] = None,
        original_width: Optional[int] = None,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):

        self.unit_runner = PipelineUnitRunnerMultiPos()

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        assert cfg_merge == False, "cfg_merge should be False"
        inputs_posi_list = [{
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }]
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        if poses is not None or cam_pose is not None or pose_video is not None or hand_pose is not None:
            pose_cond, pose_uncond = {}, {}
            if poses is not None:
                pose_cond.update({"poses": poses})
                pose_uncond.update({"poses": np.zeros_like(poses)})
            if cam_pose is not None:
                pose_cond.update({"cam_pose": cam_pose})
                pose_uncond.update({"cam_pose": np.zeros_like(cam_pose)})
            if hand_pose is not None:
                pose_cond.update({"hand_pose": hand_pose})
                pose_uncond.update({"hand_pose": np.zeros_like(hand_pose)})
            if pose_video is not None:
                first_img = pose_video[0]
                gray_array = np.full((first_img.size[1], first_img.size[0], 3), 128, dtype=np.uint8)
                zero_pose_video = [Image.fromarray(gray_array) for i in range(len(pose_video))]
                pose_cond.update({"pose_video": pose_video})
                pose_uncond.update({"pose_video": zero_pose_video})
            if camera_intrinsics is not None:
                pose_cond.update({"camera_intrinsics": camera_intrinsics})
                pose_uncond.update({"camera_intrinsics": camera_intrinsics})

            inputs_posi_list[0].update(copy.deepcopy(pose_uncond))
            inputs_nega.update(pose_uncond)
            inputs_posi_list.append({
                "prompt": prompt,
                "vap_prompt": vap_prompt,
                "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
            })
            inputs_posi_list[-1].update(pose_cond)
            
        if isinstance(cfg_scale, float):
            cfg_scale = [cfg_scale] * len(inputs_posi_list)
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "original_height": original_height, "original_width": original_width, "num_frames": num_frames,
            "hand_pose_render_mode": hand_pose_render_mode,
            "hand_pose_attr_space": hand_pose_attr_space,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, 
        }
        for unit in self.units:
            inputs_shared, inputs_posi_list, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi_list, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                in_iteration_models_2 = self.in_iteration_models_2
                if self.pose_controller2 is None:
                    in_iteration_models_2 = tuple(
                        "pose_controller" if name == "pose_controller2" else name
                        for name in in_iteration_models_2
                    )
                if self.hand_motion_controller2 is None:
                    in_iteration_models_2 = tuple(
                        "hand_motion_controller" if name == "hand_motion_controller2" else name
                        for name in in_iteration_models_2
                    )
                if self.pose_video_controller2 is None:
                    in_iteration_models_2 = tuple(
                        "pose_video_controller" if name == "pose_video_controller2" else name
                        for name in in_iteration_models_2
                    )
                if self.motion_controller2 is None:
                    in_iteration_models_2 = tuple(
                        "motion_controller" if name == "motion_controller2" else name
                        for name in in_iteration_models_2
                    )
                self.load_models_to_device(in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                if self.pose_controller2 is not None:
                    models["pose_controller"] = self.pose_controller2
                if self.hand_motion_controller2 is not None:
                    models["hand_motion_controller"] = self.hand_motion_controller2
                if self.pose_video_controller2 is not None:
                    models["pose_video_controller"] = self.pose_video_controller2
                if self.motion_controller2 is not None:
                    models["motion_controller"] = self.motion_controller2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            no_need_to_compute = cfg_scale[0] == 1.0
            noise_pred_posi_list = []
            for i, inputs_posi in enumerate(inputs_posi_list):
                no_need_to_compute = no_need_to_compute and i+1 < len(cfg_scale) and cfg_scale[i+1] == 1.0
                if no_need_to_compute:
                    noise_pred_posi = torch.zeros_like(inputs_shared["latents"])
                else:
                    noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
                noise_pred_posi_list.append(noise_pred_posi)
            if len(cfg_scale) > 1 or cfg_scale[0] != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    if cfg_scale[0] == 1.0:
                        noise_pred_nega = torch.zeros_like(inputs_shared["latents"])
                    else:
                        noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                # noise_pred = noise_pred_nega + sum([cfg_scale[i] * (noise_pred_posi - noise_pred_nega) for i, noise_pred_posi in enumerate(noise_pred_posi_list)])
                noise_pred = noise_pred_nega + cfg_scale[0] * (noise_pred_posi_list[0] - noise_pred_nega) + cfg_scale[1] * (noise_pred_posi_list[1] - noise_pred_posi_list[0])
            else:
                noise_pred = noise_pred_posi_list[0]

            # noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            # if cfg_scale != 1.0:
            #     if cfg_merge:
            #         noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
            #     else:
            #         noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            #     noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            # else:
            #     noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video

    def decode(self, latents, output_type="quantized", tiled: Optional[bool] = True,
            tile_size: Optional[tuple[int, int]] = (30, 52),
            tile_stride: Optional[tuple[int, int]] = (15, 26)):

        self.load_models_to_device(['vae'])
        video = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            output_params=("clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            output_params=("reference_latents", "clip_feature"),
            onload_model_names=("vae", "image_encoder")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("control_camera_latents_input", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}

class WanVideoUnit_RebasePoseToClipStart(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"poses": "poses", "cam_pose": "cam_pose", "hand_pose": "hand_pose"},
            input_params_nega={"poses": "poses", "cam_pose": "cam_pose", "hand_pose": "hand_pose"},
            output_params=("poses", "cam_pose", "hand_pose"),
        )

    def _clip_tensor(self, value, device, dtype):
        if value is None:
            return None
        return torch.as_tensor(value, device=device, dtype=dtype).unsqueeze(0)

    def _rebase_poses(self, poses, device, dtype):
        if poses is None:
            return None
        poses = self._clip_tensor(poses, device, dtype)
        if poses.detach().abs().amax() > 0:
            poses = poses.clone()
            poses[:, 0, -9:] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                dtype=poses.dtype,
                device=poses.device,
            )
        return poses

    def _rebase_hand_pose(self, hand_pose, clip_start_cam, device):
        if hand_pose is None:
            return None
        hand_pose = self._clip_tensor(hand_pose, device, torch.float32)
        if hand_pose.detach().abs().amax() == 0:
            return hand_pose
        rotation = clip_start_cam[:, :3, :3]
        translation = clip_start_cam[:, :3, 3]
        hand_pose = torch.einsum("bij,btvj->btvi", rotation, hand_pose) + translation[:, None, None, :]
        return hand_pose

    def process(self, pipe: WanVideoPipeline, poses, cam_pose, hand_pose):
        device = pipe.device if isinstance(pipe.device, torch.device) else torch.device(pipe.device)
        outputs = {"poses": self._rebase_poses(poses, device, pipe.torch_dtype)}
        if cam_pose is None:
            if hand_pose is not None:
                raise ValueError("cam_pose is required when hand_pose is provided.")
            return {key: value for key, value in outputs.items() if value is not None}

        cam_pose = self._clip_tensor(cam_pose, device, torch.float32)
        outputs["hand_pose"] = self._clip_tensor(hand_pose, device, torch.float32)
        if cam_pose.detach().abs().amax() == 0:
            outputs["cam_pose"] = cam_pose
            return {key: value for key, value in outputs.items() if value is not None}

        clip_start_cam = cam_pose[:, 0]
        clip_start_inv = torch.linalg.inv(clip_start_cam)
        outputs["cam_pose"] = cam_pose @ clip_start_inv[:, None]
        rebased_hand_pose = self._rebase_hand_pose(hand_pose, clip_start_cam, device)
        if rebased_hand_pose is not None:
            outputs["hand_pose"] = rebased_hand_pose
        return {key: value for key, value in outputs.items() if value is not None}
    
class WanVideoUnit_Pose2Plucker(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("height", "width", "original_height", "original_width"),
            input_params_posi={"cam_pose": "cam_pose", "camera_intrinsics": "camera_intrinsics"},
            input_params_nega={"cam_pose": "cam_pose", "camera_intrinsics": "camera_intrinsics"},
            output_params=("plucker_rays", "control_camera_valid_mask")
        )

    def process(self, pipe: WanVideoPipeline, cam_pose, camera_intrinsics, height, width, original_height, original_width):
        if cam_pose is None:
            return {}
        if original_height is not None and original_width is not None:
            o_height, o_width = int(original_height), int(original_width)
        else:
            original_sizes = (
                (540, 544),
                (540, 720),
                (540, 960),
            )
            target_aspect = width / height
            o_height, o_width = min(
                original_sizes,
                key=lambda size: abs((size[1] / size[0]) - target_aspect),
            )
        plucker_rays = poses_to_plucker_rays(
            cam_pose[0],
            orig_height=o_height,
            orig_width=o_width,
            target_height=height,
            target_width=width,
            camera_intrinsics=camera_intrinsics,
        ).unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        valid_value = float((plucker_rays[:, :, 0:1].detach().abs().amax() > 0).item())
        control_camera_valid_mask = torch.tensor(
            [valid_value], dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"plucker_rays": plucker_rays, "control_camera_valid_mask": control_camera_valid_mask}


class WanVideoUnit_PosePluckerControlAdapter(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"poses_latents": "poses_latents", "plucker_rays": "plucker_rays", "control_camera_valid_mask": "control_camera_valid_mask"},
            input_params_nega={"poses_latents": "poses_latents", "plucker_rays": "plucker_rays", "control_camera_valid_mask": "control_camera_valid_mask"},
            output_params=("control_camera_latents_input")
        )

    def process(self, pipe: WanVideoPipeline, poses_latents, plucker_rays, control_camera_valid_mask):
        if poses_latents is None or plucker_rays is None:
            return {}
        control_video = torch.cat([poses_latents, plucker_rays], dim=1)
        if control_camera_valid_mask is None:
            control_camera_valid_mask = torch.tensor(
                [1.0], dtype=control_video.dtype, device=control_video.device
            )
        if pipe.dit.control_adapter is not None and pipe.dit.in_dim_control_adapter > control_video.shape[1] * 4:
            valid_mask = control_camera_valid_mask.to(dtype=control_video.dtype, device=control_video.device)
            valid_mask = valid_mask.reshape(-1, 1, 1, 1, 1)
            valid_mask = valid_mask.expand(control_video.shape[0], 1, control_video.shape[2], control_video.shape[3], control_video.shape[4])
            control_video = torch.cat([control_video, valid_mask], dim=1)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_video[:, :, 0:1], repeats=4, dim=2),
                control_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"control_camera_latents_input": control_camera_latents_input}


class WanVideoUnit_HandPosePluckerControlAdapter(PipelineUnit):
    JOINT_OCC_PLUCKER_MODE = "joint_occ_plucker"

    def __init__(self, faces_path=None, joint_regressor_path=None):
        super().__init__(
            seperate_cfg=True,
            input_params=("original_height", "original_width", "hand_pose_render_mode", "hand_pose_attr_space", "y"),
            input_params_posi={
                "poses": "poses",
                "hand_pose": "hand_pose",
                "cam_pose": "cam_pose",
                "camera_intrinsics": "camera_intrinsics",
                "plucker_rays": "plucker_rays",
                "control_camera_valid_mask": "control_camera_valid_mask",
            },
            input_params_nega={
                "poses": "poses",
                "hand_pose": "hand_pose",
                "cam_pose": "cam_pose",
                "camera_intrinsics": "camera_intrinsics",
                "plucker_rays": "plucker_rays",
                "control_camera_valid_mask": "control_camera_valid_mask",
            },
            output_params=("control_camera_latents_input", "hand_motion_masks_input")
        )
        self.faces_path = faces_path or self._find_default_faces_path()
        self.faces_np = None
        if self.faces_path is not None and os.path.exists(self.faces_path):
            self.faces_np = np.loadtxt(self.faces_path, dtype=np.int64)
        self.joint_feature_builder = JointOccPluckerFeatureBuilder(joint_regressor_path)
        self._dr = None
        self._ctx = None
        self._ctx_device = None

    def _pipe_device(self, pipe):
        return pipe.device if isinstance(pipe.device, torch.device) else torch.device(pipe.device)

    def _find_default_faces_path(self):
        candidates = [
            os.environ.get("HAND_MESH_FACES_PATH"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _get_context(self, device):
        device = device if isinstance(device, torch.device) else torch.device(device)
        if self._ctx is not None and self._ctx_device == device:
            return self._dr, self._ctx
        if device.type != "cuda":
            raise RuntimeError("Hand pose rasterization requires a CUDA device.")

        import nvdiffrast.torch as dr

        try:
            ctx = dr.RasterizeCudaContext(device=device)
        except TypeError:
            ctx = dr.RasterizeCudaContext()
        self._dr = dr
        self._ctx = ctx
        self._ctx_device = device
        return dr, ctx

    def _get_faces(self, device):
        if self.faces_np is None:
            raise FileNotFoundError(
                "Could not find hand mesh faces. Set HAND_MESH_FACES_PATH to your mesh faces file."
            )
        return torch.from_numpy(self.faces_np).to(device=device, dtype=torch.int32)

    def _to_float_tensor(self, value, device):
        if not torch.is_tensor(value):
            value = torch.from_numpy(value)
        return value.to(device=device, dtype=torch.float32)

    def _camera_intrinsics(self, height, width, original_height, original_width, device, camera_intrinsics=None):
        if original_height is not None and original_width is not None:
            o_height, o_width = int(original_height), int(original_width)
        else:
            original_sizes = (
                (540, 544),
                (540, 720),
                (540, 960),
            )
            target_aspect = width / height
            o_height, o_width = min(
                original_sizes,
                key=lambda size: abs((size[1] / size[0]) - target_aspect),
            )

        if camera_intrinsics is not None:
            K_orig = self._to_float_tensor(camera_intrinsics, device)
            if K_orig.ndim == 3:
                K_orig = K_orig[0]
            if K_orig.shape != (3, 3):
                raise ValueError(f"camera_intrinsics must have shape [3, 3], got {tuple(K_orig.shape)}")
        elif o_height == 1080:
            K_orig = taco_1080p_intrinsics(o_height, o_width, device)
        else:
            K_orig = torch.tensor([
                [600.0, 0.0, o_width / 2],
                [0.0, 600.0, o_height / 2],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32, device=device)
        return adjust_intrinsics_for_resize(K_orig, o_height, o_width, height, width)

    def _projection_matrix(self, width, height, K, znear, zfar, device):
        if not torch.isclose(K[0, 1], torch.zeros((), dtype=K.dtype, device=K.device)):
            raise ValueError("Hand rasterization does not support camera intrinsics with non-zero skew.")
        return torch.tensor(
            [
                [2 * K[0, 0] / width, 0.0, 0.0, 0.0],
                [0.0, -2 * K[1, 1] / height, 0.0, 0.0],
                [
                    (width - 2 * K[0, 2]) / width,
                    (height - 2 * K[1, 2]) / height,
                    -(zfar + znear) / (zfar - znear),
                    -1.0,
                ],
                [0.0, 0.0, -2 * zfar * znear / (zfar - znear), 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ).transpose(0, 1)

    def _vertex_normals(self, vertices, faces):
        tris = vertices[:, faces.long()]
        face_normals = torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0], dim=-1)
        normals = torch.zeros_like(vertices)
        for corner in range(3):
            normals.index_add_(1, faces[:, corner].long(), face_normals)
        return torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)

    def _hand_identity_colors(self, num_vertices, device):
        left_color = torch.tensor([0.804, 0.600, 0.820], dtype=torch.float32, device=device)
        right_color = torch.tensor([0.207, 0.596, 0.792], dtype=torch.float32, device=device)
        colors = right_color.reshape(1, 3).expand(num_vertices, 3).clone()
        if num_vertices >= 1556:
            colors[:778] = left_color
            colors[778:1556] = right_color
        elif num_vertices > 778:
            split = num_vertices // 2
            colors[:split] = left_color
            colors[split:] = right_color
        return colors

    def _pose_attr_channels(self, mode):
        if mode in ("hand_mask", "pose9d_mask_broadcast"):
            return 3
        if mode in ("normal_plucker", "reflection_plucker", "reflective_rays_plucker"):
            return 6
        return 3

    def _split_hand_pose9d(self, poses):
        if poses is None:
            raise ValueError("pose9d_mask_broadcast requires poses with left16x9 + right16x9 + camera9.")
        left_pose = poses[..., :16 * 9]
        right_pose = poses[..., 16 * 9:2 * 16 * 9]
        return left_pose, right_pose

    def _hand_side_attr(self, num_vertices, device):
        side = torch.ones((num_vertices, 1), dtype=torch.float32, device=device)
        if num_vertices >= 1556:
            side[:778] = -1.0
            side[778:1556] = 1.0
        elif num_vertices > 778:
            split = num_vertices // 2
            side[:split] = -1.0
            side[split:] = 1.0
        return side

    def _render_pose_latents(
        self,
        hand_pose,
        cam_pose,
        mode,
        attr_space,
        height,
        width,
        original_height,
        original_width,
        batch_size,
        device,
        poses=None,
        camera_intrinsics=None,
    ):
        render_hand_masks = mode in ("hand_mask", "pose9d_mask_broadcast")
        is_zero = hand_pose.detach().abs().amax() == 0 or cam_pose.detach().abs().amax() == 0
        if is_zero:
            return torch.zeros(
                (hand_pose.shape[0], self._pose_attr_channels(mode), hand_pose.shape[1], height, width),
                dtype=torch.float32,
                device=device,
            )

        dr, ctx = self._get_context(device)
        faces = self._get_faces(device)
        # nvdiffrast's CUDA rasterizer/interpolate/antialias require float32 inputs.
        # Under accelerate mixed precision the enclosing forward runs inside an
        # autocast region that would silently downcast the geometry matmuls to
        # bf16/fp16, so disable autocast for the whole (cheap) rendering block.
        device_type = torch.device(device).type
        with torch.autocast(device_type=device_type, enabled=False):
            K = self._camera_intrinsics(height, width, original_height, original_width, device, camera_intrinsics)
            proj = self._projection_matrix(width, height, K, 1e-3, 1e3, device)
            cv_to_gl = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )

            rendered = []
            num_frames = min(hand_pose.shape[1], cam_pose.shape[1])
            for batch_id in range(hand_pose.shape[0]):
                frames = []
                vertices = hand_pose[batch_id, :num_frames]
                camera = cam_pose[batch_id, :num_frames]
                if render_hand_masks:
                    side_attr = self._hand_side_attr(vertices.shape[1], device).unsqueeze(0)
                    normals = None
                else:
                    normals = self._vertex_normals(vertices, faces)
                for start in range(0, num_frames, batch_size):
                    end = min(start + batch_size, num_frames)
                    cur_vertices = vertices[start:end]
                    cur_camera = camera[start:end]
                    cur_b = end - start

                    ones = torch.ones((cur_b, cur_vertices.shape[1], 1), dtype=torch.float32, device=device)
                    pos_h = torch.cat([cur_vertices, ones], dim=-1)

                    mv = cv_to_gl.unsqueeze(0) @ cur_camera
                    mvp = proj.unsqueeze(0) @ mv
                    pos_clip = torch.matmul(pos_h, mvp.transpose(1, 2)).contiguous()
                    rast, _ = dr.rasterize(ctx, pos_clip, faces, resolution=[height, width])

                    if render_hand_masks:
                        side_feat, _ = dr.interpolate(
                            side_attr.expand(cur_b, -1, -1).contiguous(),
                            rast,
                            faces,
                        )
                        alpha = (rast[..., 3:4] > 0).to(side_feat.dtype)
                        left_mask = (side_feat < 0).to(side_feat.dtype) * alpha
                        right_mask = (side_feat > 0).to(side_feat.dtype) * alpha
                        hand_mask = torch.clamp(left_mask + right_mask, 0.0, 1.0)
                        feat = torch.cat(
                            [
                                left_mask.permute(0, 3, 1, 2),
                                right_mask.permute(0, 3, 1, 2),
                                hand_mask.permute(0, 3, 1, 2),
                            ],
                            dim=1,
                        )
                        frames.append(feat)
                        continue

                    cur_normals = normals[start:end]
                    pos_cam_h = torch.matmul(pos_h, cur_camera.transpose(1, 2))
                    pos_cam = pos_cam_h[..., :3]
                    normal_cam = torch.einsum("bij,bvj->bvi", cur_camera[:, :3, :3], cur_normals)
                    normal_cam = torch.nn.functional.normalize(normal_cam, dim=-1, eps=1e-6)

                    if attr_space == "camera":
                        attr_position = pos_cam
                        attr_normal = normal_cam
                        attr_view_dir = torch.nn.functional.normalize(pos_cam, dim=-1, eps=1e-6)
                    elif attr_space == "world":
                        attr_position = cur_vertices
                        attr_normal = cur_normals
                        rotation = cur_camera[:, :3, :3]
                        translation = cur_camera[:, :3, 3]
                        camera_center = -torch.einsum("bij,bj->bi", rotation.transpose(1, 2), translation)
                        attr_view_dir = torch.nn.functional.normalize(
                            cur_vertices - camera_center[:, None, :], dim=-1, eps=1e-6
                        )
                    else:
                        raise ValueError(f"Unsupported hand_pose_attr_space: {attr_space}")

                    if mode == "position":
                        attr = attr_position
                    elif mode == "depth":
                        attr = attr_position[..., 2:3].expand(-1, -1, 3)
                    elif mode == "normal" or mode == "surface_normal":
                        attr = attr_normal
                    elif mode == "reflection" or mode == "reflective_rays":
                        attr = attr_view_dir - 2.0 * (attr_view_dir * attr_normal).sum(dim=-1, keepdim=True) * attr_normal
                        attr = torch.nn.functional.normalize(attr, dim=-1, eps=1e-6)
                    elif mode == "silhouette_wireframe":
                        attr = self._hand_identity_colors(cur_vertices.shape[1], device).unsqueeze(0).expand(cur_b, -1, -1)
                    elif mode == "normal_plucker":
                        attr = surface_directions(attr_normal, attr_view_dir, "normal_plucker")
                    elif mode == "reflection_plucker" or mode == "reflective_rays_plucker":
                        attr = surface_directions(attr_normal, attr_view_dir, "reflection_plucker")
                    else:
                        raise ValueError(f"Unsupported hand_pose_render_mode: {mode}")

                    feat, _ = dr.interpolate(attr.contiguous(), rast, faces)
                    alpha = (rast[..., 3:4] > 0).to(feat.dtype)
                    if mode in ("normal_plucker", "reflection_plucker", "reflective_rays_plucker"):
                        pos_feat, _ = dr.interpolate(attr_position.contiguous(), rast, faces)
                        feat = plucker_from_direction(pos_feat, feat)
                    elif mode == "silhouette_wireframe":
                        bary = torch.stack(
                            [rast[..., 0], rast[..., 1], 1.0 - rast[..., 0] - rast[..., 1]],
                            dim=-1,
                        )
                        edge_dist = bary.min(dim=-1, keepdim=True).values
                        wire_alpha = (edge_dist < 0.025).to(feat.dtype) * alpha
                        fill = feat * 0.45
                        wire = torch.clamp(feat * 1.25 + 0.15, 0.0, 1.0)
                        feat = fill * (1.0 - wire_alpha) + wire * wire_alpha
                    feat = torch.cat([feat * alpha, alpha], dim=-1)
                    feat = dr.antialias(feat.contiguous(), rast, pos_clip, faces)[..., :self._pose_attr_channels(mode)]
                    frames.append(feat.permute(0, 3, 1, 2))
                rendered.append(torch.cat(frames, dim=0).permute(1, 0, 2, 3))
            return torch.stack(rendered, dim=0)

    def _pack_temporal_video(self, pipe, control_video):
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_video[:, :, 0:1], repeats=4, dim=2),
                control_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        return control_camera_latents.to(device=self._pipe_device(pipe), dtype=pipe.torch_dtype)

    def _pack_control_video(self, pipe, control_video, control_camera_valid_mask):
        if control_camera_valid_mask is None:
            control_camera_valid_mask = torch.tensor(
                [1.0], dtype=control_video.dtype, device=control_video.device
            )
        if pipe.dit.control_adapter is not None and pipe.dit.in_dim_control_adapter > control_video.shape[1] * 4:
            valid_mask = control_camera_valid_mask.to(dtype=control_video.dtype, device=control_video.device)
            valid_mask = valid_mask.reshape(-1, 1, 1, 1, 1)
            valid_mask = valid_mask.expand(
                control_video.shape[0], 1, control_video.shape[2],
                control_video.shape[3], control_video.shape[4]
            )
            control_video = torch.cat([control_video, valid_mask], dim=1)
        return self._pack_temporal_video(pipe, control_video)

    def _render_joint_occ_plucker_latents(
        self,
        pipe,
        hand_pose,
        cam_pose,
        camera_intrinsics,
        plucker_rays,
        original_height,
        original_width,
        y,
        device,
    ):
        packed_plucker = self._pack_temporal_video(pipe, plucker_rays.float())
        grid_height, grid_width = plucker_rays.shape[-2] // 8, plucker_rays.shape[-1] // 8
        packed_plucker = resize_packed_plucker_to_latent_grid(packed_plucker, grid_height, grid_width)

        if hand_pose.detach().abs().amax() == 0 or cam_pose.detach().abs().amax() == 0:
            joint_latents = torch.zeros(
                (hand_pose.shape[0], 80, packed_plucker.shape[2], grid_height, grid_width),
                dtype=torch.float32,
                device=device,
            )
        else:
            K = self._camera_intrinsics(
                plucker_rays.shape[-2],
                plucker_rays.shape[-1],
                original_height,
                original_width,
                device,
                camera_intrinsics,
            )
            z0 = self.joint_feature_builder.extract_reference_vae_latent(
                y,
                hand_pose.shape[0],
                grid_height,
                grid_width,
                device,
            )
            joint_latents = self.joint_feature_builder.build_joint_occ_latents(
                hand_pose,
                cam_pose,
                K,
                z0,
                grid_height,
                grid_width,
                device,
            )

        if joint_latents.shape[2] != packed_plucker.shape[2]:
            num_frames = min(joint_latents.shape[2], packed_plucker.shape[2])
            joint_latents = joint_latents[:, :, :num_frames]
            packed_plucker = packed_plucker[:, :, :num_frames]
        control_camera_latents_input = torch.cat([packed_plucker, joint_latents], dim=1)
        expected_dim = getattr(pipe.dit, "in_dim_control_adapter", None)
        if expected_dim is not None and expected_dim != control_camera_latents_input.shape[1]:
            raise ValueError(
                f"{self.JOINT_OCC_PLUCKER_MODE} produces {control_camera_latents_input.shape[1]} control channels "
                f"(24 packed plucker + 80 joint), but the loaded control adapter expects {expected_dim}. "
                "Create/load an adapter with --in_dim 104 --downscale_factor 1."
            )
        return control_camera_latents_input.to(device=device, dtype=pipe.torch_dtype)

    def process(
        self,
        pipe: WanVideoPipeline,
        poses,
        hand_pose,
        cam_pose,
        camera_intrinsics,
        plucker_rays,
        control_camera_valid_mask,
        original_height,
        original_width,
        hand_pose_render_mode,
        hand_pose_attr_space,
        y,
    ):
        if hand_pose is None or cam_pose is None or plucker_rays is None:
            return {}
        render_device = self._pipe_device(pipe)
        hand_pose = self._to_float_tensor(hand_pose, render_device)
        cam_pose = self._to_float_tensor(cam_pose, render_device)
        plucker_rays = self._to_float_tensor(plucker_rays, render_device)
        if camera_intrinsics is not None:
            camera_intrinsics = self._to_float_tensor(camera_intrinsics, render_device)
        if poses is not None:
            poses = self._to_float_tensor(poses, render_device)
        render_height, render_width = plucker_rays.shape[-2], plucker_rays.shape[-1]

        if hand_pose_render_mode == self.JOINT_OCC_PLUCKER_MODE:
            control_camera_latents_input = self._render_joint_occ_plucker_latents(
                pipe,
                hand_pose,
                cam_pose,
                camera_intrinsics,
                plucker_rays,
                original_height,
                original_width,
                y,
                render_device,
            )
            return {
                "control_camera_latents_input": control_camera_latents_input,
                "hand_motion_masks_input": None,
            }

        with torch.no_grad():
            pose_latents = self._render_pose_latents(
                hand_pose,
                cam_pose,
                hand_pose_render_mode,
                hand_pose_attr_space,
                render_height,
                render_width,
                original_height,
                original_width,
                batch_size=128,
                device=render_device,
                poses=poses,
                camera_intrinsics=camera_intrinsics,
            )
        control_video = combine_control_maps(pose_latents, plucker_rays)
        control_camera_latents_input = self._pack_control_video(
            pipe, control_video, control_camera_valid_mask
        )
        if hand_pose_render_mode in ("hand_mask", "pose9d_mask_broadcast"):
            hand_motion_masks_input = self._pack_temporal_video(pipe, pose_latents[:, :2])
        else:
            hand_motion_masks_input = None
        return {
            "control_camera_latents_input": control_camera_latents_input,
            "hand_motion_masks_input": hand_motion_masks_input,
        }


class WanVideoUnit_PoseVideoControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"pose_video": "pose_video", "tiled": "tiled", "tile_size": "tile_size", "tile_stride": "tile_stride"},
            input_params_nega={"pose_video": "pose_video", "tiled": "tiled", "tile_size": "tile_size", "tile_stride": "tile_stride"},
            output_params=("poses_latents",)
        )

    def process(self, pipe: WanVideoPipeline, pose_video, tiled, tile_size, tile_stride):
        if pose_video is None:
            return {}
        pose_video = pipe.preprocess_video(pose_video)
        poses_latents = pose_video
        # poses_latents = pipe.vae.encode(pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"poses_latents": poses_latents}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",),
            output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_VAP(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder", "vae", "image_encoder"),
            input_params=("vap_video", "vap_prompt", "negative_vap_prompt", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("vap_clip_feature", "vap_hidden_state", "context_vap")
        )
        
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("vap_video") is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            # 1. encode vap prompt
            pipe.load_models_to_device(["text_encoder"])
            vap_prompt, negative_vap_prompt = inputs_posi.get("vap_prompt", ""), inputs_nega.get("negative_vap_prompt", "")
            vap_prompt_emb = self.encode_prompt(pipe, vap_prompt)
            negative_vap_prompt_emb = self.encode_prompt(pipe, negative_vap_prompt)
            inputs_posi.update({"context_vap":vap_prompt_emb})
            inputs_nega.update({"context_vap":negative_vap_prompt_emb})
            # 2. prepare vap image clip embedding
            pipe.load_models_to_device(["vae", "image_encoder"])
            vap_video, end_image = inputs_shared.get("vap_video"), inputs_shared.get("end_image")

            num_frames, height, width = inputs_shared.get("num_frames"),inputs_shared.get("height"), inputs_shared.get("width")
            
            image_vap = pipe.preprocess_image(vap_video[0].resize((width, height))).to(pipe.device)

            vap_clip_context = pipe.image_encoder.encode_image([image_vap])
            if end_image is not None:
                vap_end_image = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                if pipe.dit.has_image_pos_emb:
                    vap_clip_context = torch.concat([vap_clip_context, pipe.image_encoder.encode_image([vap_end_image])], dim=1)
            vap_clip_context = vap_clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_clip_feature":vap_clip_context})

            # 3. prepare vap latents            
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            if end_image is not None:
                msk[:, -1:] = 1
                last_image_vap = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                vae_input = torch.concat([image_vap.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image_vap.device), last_image_vap.transpose(0,1)],dim=1)
            else:
                vae_input = torch.concat([image_vap.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image_vap.device)], dim=1)
            
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]

            tiled,tile_size,tile_stride = inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")

            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_video = pipe.preprocess_video(vap_video)
            vap_latent = pipe.vae.encode(vap_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_latent = torch.concat([vap_latent,y], dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_hidden_state":vap_latent})

            return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",),
            input_params=("input_audio", "audio_embeds", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "audio_sample_rate", "s2v_pose_video", "s2v_pose_latents", "motion_video"),
            output_params=("audio_embeds", "motion_latents", "drop_motion_frames", "s2v_pose_latents"),
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None:
            assert motion_video.shape[2] == motion_frames, f"motion video must have {motion_frames} frames, but got {motion_video.shape[2]}"
            motion_latents = motion_video
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"),
            output_params=("animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video")
        )

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            output_params=("pose_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("animate_face_video",),
            output_params=("face_pixel_values"),
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class WanVideoUnit_LongCatVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("longcat_video",),
            output_params=("longcat_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, longcat_video):
        if longcat_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        longcat_video = pipe.preprocess_video(longcat_video)
        longcat_latents = pipe.vae.encode(longcat_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"longcat_latents": longcat_latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    pose_controller: WanPoseControllerModel = None,
    hand_motion_controller: WanHandMotionTokenEncoder = None,
    pose_video_controller: Union[WanPoseVideoControllerModel, SimpleAdapter] = None,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state = None,
    vap_clip_feature = None,
    context_vap = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    poses: Optional[torch.Tensor] = None,
    poses_latents: Optional[torch.Tensor] = None,
    plucker_rays: Optional[torch.Tensor] = None,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    control_camera_valid_mask = None,
    hand_motion_masks_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    
    if control_camera_valid_mask is None:
        control_camera_valid_mask = torch.tensor(
            [1.0], dtype=latents.dtype, device=latents.device
        )
    # print(poses.shape, poses_latents.shape, plucker_rays.shape)
    # print("==========")
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            pose_controller=pose_controller,
            hand_motion_controller=hand_motion_controller,
            pose_video_controller=pose_video_controller,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            poses=poses,
            poses_latents=poses_latents,
            plucker_rays=plucker_rays,
            control_camera_latents_input=control_camera_latents_input,
            control_camera_valid_mask=control_camera_valid_mask,
            hand_motion_masks_input=hand_motion_masks_input,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # LongCat-Video
    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    nvtx_push("wan_model_fn/pre_dit_blocks")
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Pose Controller
    if poses is not None and pose_controller is not None:
        # import ipdb; ipdb.set_trace()
        assert use_unified_sequence_parallel is False, "usp not support yet"
        if len(t_mod.shape) == 3:
            t_mod = t_mod.unsqueeze(1)
            e_t = t.unsqueeze(1)
        else:
            e_t = t.unflatten(1, (latents.shape[2], -1))[:, :, 0]
        t_mod = t_mod + pose_controller(poses, e_t).unflatten(2, (6, dit.dim)).unsqueeze(2).expand(-1, -1, latents.shape[3] * latents.shape[4] // 4, -1, -1).flatten(1, 2) * control_camera_valid_mask

    hand_motion_tokens = hand_motion_controller(poses) if hand_motion_controller is not None and poses is not None else None
    poses_y = None

    # Pose Video Controller
    if (
        pose_video_controller is not None
        and isinstance(pose_video_controller, SimpleAdapter)
    ):
        # assert use_unified_sequence_parallel is False, "usp not support yet"
        if control_camera_latents_input is not None:
            poses_y = pose_video_controller(control_camera_latents_input) * control_camera_valid_mask
    elif poses_latents is not None and pose_video_controller is not None:
        assert use_unified_sequence_parallel is False, "usp not support yet"
        if plucker_rays is not None:
            poses_latents = torch.cat([poses_latents, plucker_rays], dim=1)
        if len(t_mod.shape) == 3:
            t_mod = t_mod.unsqueeze(1)
            e_t = t.unsqueeze(1) # [B, 1, D]
        else:
            e_t = t # [B, TxHxW, D]
        t_mod = t_mod + pose_video_controller(poses_latents, e_t).unflatten(2, (6, dit.dim)) * control_camera_valid_mask
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if poses_y is not None:
        if poses_y.shape[2:] != x.shape[2:]:
            raise ValueError(
                "pose_video_controller latent adapter output shape %s does not match latents shape %s"
                % (tuple(poses_y.shape), tuple(x.shape))
            )
        x = torch.cat([x, poses_y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        
    # Camera control
    x = dit.patchify(
        x,
        control_camera_latents_input,
        control_camera_valid_mask,
        hand_motion_tokens=hand_motion_tokens,
        hand_motion_masks_input=hand_motion_masks_input,
    )
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # VAP 
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f,h,w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    nvtx_pop()
    
    # blocks
    nvtx_push("wan_model_fn/dit_blocks")
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs)
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
    nvtx_pop()

    nvtx_push("wan_model_fn/head_unpatchify")
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    nvtx_pop()
    return x


def model_fn_longcat_video(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    context = context.unsqueeze(0)
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        num_cond_latents=num_cond_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    output = -output
    output = output.to(latents.dtype)
    return output


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
