# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image
import numpy as np
import decord


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data, **kwargs):
        for operator in self.operators:
            data = operator(data, **kwargs)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data, **kwargs):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data, **kwargs):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str, **kwargs):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        width, height = image.size
        if self.height is None and self.width is None:
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        elif self.height is None:
            scale = self.width / width
            height = round(height * scale / self.height_division_factor) * self.height_division_factor
            width = self.width
        elif self.width is None:
            scale = self.height / height
            width = round(width * scale / self.width_division_factor) * self.width_division_factor
            height = self.height
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image, **kwargs):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ImageResize(ImageCropAndResize):
    """Resize the complete image to the target size without cropping it."""

    def crop_and_resize(self, image, target_height, target_width):
        return torchvision.transforms.functional.resize(
            image,
            (target_height, target_width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )


class VideoFrames(list):
    """Decoded frames carrying the source video's ``(height, width)``."""

    def __init__(self, frames=(), original_size=None):
        super().__init__(frames)
        self.original_size = original_size


def resolve_original_video_size(frames, declared_height=None, declared_width=None):
    """Return native ``(height, width)`` and reject stale manifest dimensions."""

    def optional_int(value):
        if value is None:
            return None
        try:
            if bool(np.isnan(value)):
                return None
        except (TypeError, ValueError):
            pass
        return int(value)

    declared = (optional_int(declared_height), optional_int(declared_width))
    decoded = getattr(frames, "original_size", None)
    if decoded is None:
        return declared
    decoded = (int(decoded[0]), int(decoded[1]))
    for name, expected, actual in zip(("height", "width"), declared, decoded):
        if expected is not None and expected != actual:
            raise ValueError(
                f"Manifest video {name} ({expected}) does not match decoded {name} ({actual})."
            )
    return decoded


class ToList(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, total_frames):
        num_frames = self.num_frames
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, start_frame=0, **kwargs):
        vr = decord.VideoReader(data, num_threads=1)
        start_frame = int(start_frame)
        remaining_frames = max(0, len(vr) - start_frame)
        num_frames = self.get_num_frames(remaining_frames)
        if num_frames <= 0:
            return []
        
        # 使用 decord 批量读取帧（更高效）
        indices = list(range(start_frame, start_frame + num_frames))
        frames_np = vr.get_batch(indices).asnumpy()  # (T, H, W, C) numpy array
        
        # 转换为 PIL Image 并应用 frame_processor
        original_size = None
        if frames_np.ndim == 4 and frames_np.shape[0] > 0:
            original_size = (int(frames_np.shape[1]), int(frames_np.shape[2]))
        frames = VideoFrames(original_size=original_size)
        for frame_np in frames_np:
            frame = Image.fromarray(frame_np)
            frame = self.frame_processor(frame)
            frames.append(frame)
        
        return frames
        
    # def __call__(self, data: str):
    #     reader = imageio.get_reader(data)
    #     num_frames = self.get_num_frames(reader)
    #     frames = []
    #     for frame_id in range(num_frames):
    #         frame = reader.get_data(frame_id)
    #         frame = Image.fromarray(frame)
    #         frame = self.frame_processor(frame)
    #         frames.append(frame)
    #     reader.close()
    #     return frames
    
class LoadNpy(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        
    def __call__(self, data: str, start_frame=0, **kwargs):
        frames = np.load(data)
        start_frame = int(start_frame)
        num_frames = self.num_frames
        remaining_frames = max(0, frames.shape[0] - start_frame)
        if remaining_frames < num_frames:
            num_frames = remaining_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[start_frame:start_frame + num_frames]
        return frames


class LoadNpyFile(DataProcessingOperator):
    def __call__(self, data: str, **kwargs):
        return np.load(data)


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data, **kwargs):
        return [self.operator(i, **kwargs) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, total_frames):
        num_frames = self.num_frames
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, start_frame=0, **kwargs):
        frames = []
        images = iio.imread(data, mode="RGB")
        start_frame = int(start_frame)
        remaining_frames = max(0, len(images) - start_frame)
        num_frames = self.get_num_frames(remaining_frames)
        for img in images[start_frame:start_frame + num_frames]:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str, **kwargs):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data, **kwargs)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data, **kwargs):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data, **kwargs)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data, **kwargs):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data, **kwargs):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str, **kwargs):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio
