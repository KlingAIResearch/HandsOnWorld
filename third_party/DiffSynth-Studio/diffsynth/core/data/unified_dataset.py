# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
from .operators import *
import torch, json, pandas
import random


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        random_drop_keys=tuple(),
        random_start_frame=False,
        sample_num_frames=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.random_drop_keys = random_drop_keys
        self.random_start_frame = random_start_frame
        self.sample_num_frames = sample_num_frames
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
        self.check_random_start_frame_config()
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        resize_mode="crop",
    ):
        if resize_mode == "crop":
            frame_processor_cls = ImageCropAndResize
        elif resize_mode == "resize":
            frame_processor_cls = ImageResize
        else:
            raise ValueError(f"Unsupported resize_mode: {resize_mode!r}")
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> frame_processor_cls(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor_cls(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor_cls(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("npy", "npz"), LoadNpy(num_frames, time_division_factor, time_division_remainder))
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def is_missing_value(self, value):
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        return bool(pandas.isna(value))

    def allow_random_start_for_row(self, data):
        for key in ("allow_random_start_frame", "can_random_start_frame"):
            if key in data and not self.is_missing_value(data[key]):
                value = data[key]
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "y")
                return bool(value)
        return True

    def check_random_start_frame_config(self):
        if not self.random_start_frame:
            return
        if self.load_from_cache:
            raise ValueError("random_start_frame=True requires metadata with a num_frames column.")
        if self.sample_num_frames is None:
            raise ValueError("random_start_frame=True requires sample_num_frames.")
        for data_id, data in enumerate(self.data):
            if not self.allow_random_start_for_row(data):
                continue
            if "num_frames" not in data or self.is_missing_value(data["num_frames"]):
                raise ValueError(f"random_start_frame=True requires num_frames for metadata row {data_id}.")

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            operator_kwargs = {}
            if self.random_start_frame and self.allow_random_start_for_row(data):
                max_start_frame = int(data["num_frames"]) - self.sample_num_frames
                if max_start_frame < 0:
                    raise ValueError(
                        f"num_frames ({data['num_frames']}) must be >= sample_num_frames ({self.sample_num_frames})."
                    )
                operator_kwargs["start_frame"] = random.randint(0, max_start_frame)
            for key in self.data_file_keys:
                if key in data:
                    if self.is_missing_value(data[key]):
                        data.pop(key)
                        continue
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key], **operator_kwargs)
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key], **operator_kwargs)

        pose_drop = random.random() < 0.1
        for key in self.random_drop_keys:
            drop = random.random() < 0.1 if "pose" not in key else pose_drop
            if key in data and drop:
                if isinstance(data[key], np.ndarray):
                    data[key] *= 0.
                elif isinstance(data[key], str):
                    data[key] = ""
                elif isinstance(data[key], list) and isinstance(data[key][0], Image.Image):
                    # 将列表中的所有PIL图像置为灰色 [128, 128, 128]
                    # 批量创建灰色图像
                    first_img = data[key][0]
                    gray_array = np.full((first_img.size[1], first_img.size[0], 3), 128, dtype=np.uint8)
                    data[key] = [Image.fromarray(gray_array) for i in range(len(data[key]))]
                else:
                    assert False, f"Unsupported data type for random drop: {key, type(data[key])}"
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
