# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import os, torch, queue, threading
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def nvtx_push(name):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result = torch.cuda.cudart().cudaProfilerStart()
        if result not in (0, None):
            raise RuntimeError(f"cudaProfilerStart failed with error code {result}")


def cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result = torch.cuda.cudart().cudaProfilerStop()
        if result not in (0, None):
            raise RuntimeError(f"cudaProfilerStop failed with error code {result}")


class OnlineUnitPrefetcher:
    _sentinel = object()

    def __init__(self, model: DiffusionTrainingModule, data_iter, queue_size: int = 2):
        self.model = model
        self.data_iter = data_iter
        self.queue = queue.Queue(maxsize=max(1, queue_size))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.device = torch.device(model.pipe.device)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.thread.start()

    def _record_stream(self, value, stream):
        if torch.is_tensor(value):
            value.record_stream(stream)
        elif isinstance(value, tuple):
            for item in value:
                self._record_stream(item, stream)
        elif isinstance(value, list):
            for item in value:
                self._record_stream(item, stream)
        elif isinstance(value, dict):
            for item in value.values():
                self._record_stream(item, stream)

    def _prepare(self, data):
        nvtx_push("online_prefetch/prepare_inputs")
        with torch.no_grad():
            try:
                if self.stream is None:
                    return self.model.prepare_inputs(data), None
                with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                    inputs = self.model.prepare_inputs(data)
                    event = torch.cuda.Event()
                    event.record(self.stream)
                return inputs, event
            finally:
                nvtx_pop()

    def _put(self, item):
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _worker_loop(self):
        try:
            for data in self.data_iter:
                if self.stop_event.is_set():
                    break
                self._put(self._prepare(data))
        except BaseException as error:
            self._put(error)
        finally:
            self._put(self._sentinel)

    def result(self):
        nvtx_push("online_prefetch/queue_wait")
        try:
            item = self.queue.get()
        finally:
            nvtx_pop()
        if item is self._sentinel:
            return None
        if isinstance(item, BaseException):
            raise item
        inputs, event = item
        if event is not None:
            event.wait()
            self._record_stream(inputs, torch.cuda.current_stream(self.device))
        return inputs

    def close(self):
        self.stop_event.set()
        self.thread.join()


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        start_epoch = getattr(args, "resume_start_epoch", 0)
        start_step = getattr(args, "resume_start_step", 0)
        online_unit_prefetch = getattr(args, "online_unit_prefetch", False)
        online_unit_prefetch_queue_size = getattr(args, "online_unit_prefetch_queue_size", 2)
        cuda_profiler_start_step = getattr(args, "cuda_profiler_start_step", None)
        cuda_profiler_stop_step = getattr(args, "cuda_profiler_stop_step", None)
        nvtx_capture_start_step = getattr(args, "nvtx_capture_start_step", None)
        nvtx_capture_stop_step = getattr(args, "nvtx_capture_stop_step", None)
        nvtx_capture_name = getattr(args, "nvtx_capture_name", "nsys_capture")
    else:
        start_epoch = 0
        start_step = 0
        online_unit_prefetch = False
        online_unit_prefetch_queue_size = 2
        cuda_profiler_start_step = None
        cuda_profiler_stop_step = None
        nvtx_capture_start_step = None
        nvtx_capture_stop_step = None
        nvtx_capture_name = "nsys_capture"

    release_cache_before_backward = getattr(args, "release_cache_before_backward", False)
    cuda_profiler_enabled = cuda_profiler_start_step is not None or cuda_profiler_stop_step is not None
    cuda_profiler_active = False
    if cuda_profiler_enabled:
        if cuda_profiler_start_step is None or cuda_profiler_stop_step is None:
            raise ValueError("--cuda_profiler_start_step and --cuda_profiler_stop_step must be set together.")
        if cuda_profiler_start_step < 0 or cuda_profiler_stop_step <= cuda_profiler_start_step:
            raise ValueError("--cuda_profiler_stop_step must be greater than --cuda_profiler_start_step >= 0.")
        if accelerator.is_main_process:
            print(f"CUDA profiler capture window: steps [{cuda_profiler_start_step}, {cuda_profiler_stop_step})")

    nvtx_capture_enabled = nvtx_capture_start_step is not None or nvtx_capture_stop_step is not None
    nvtx_capture_active = False
    if nvtx_capture_enabled:
        if nvtx_capture_start_step is None or nvtx_capture_stop_step is None:
            raise ValueError("--nvtx_capture_start_step and --nvtx_capture_stop_step must be set together.")
        if nvtx_capture_start_step < 0 or nvtx_capture_stop_step <= nvtx_capture_start_step:
            raise ValueError("--nvtx_capture_stop_step must be greater than --nvtx_capture_start_step >= 0.")
        if accelerator.is_main_process:
            print(f"NVTX capture window '{nvtx_capture_name}': steps [{nvtx_capture_start_step}, {nvtx_capture_stop_step})")

    def maybe_start_cuda_profiler(step):
        nonlocal cuda_profiler_active
        if cuda_profiler_enabled and not cuda_profiler_active and step == cuda_profiler_start_step:
            if accelerator.is_main_process:
                print(f"Starting CUDA profiler at step {step}.")
            cuda_profiler_start()
            cuda_profiler_active = True

    def maybe_stop_cuda_profiler(step):
        nonlocal cuda_profiler_active
        if cuda_profiler_enabled and cuda_profiler_active and step + 1 >= cuda_profiler_stop_step:
            if accelerator.is_main_process:
                print(f"Stopping CUDA profiler after step {step}.")
            cuda_profiler_stop()
            cuda_profiler_active = False

    def maybe_start_nvtx_capture(step):
        nonlocal nvtx_capture_active
        if nvtx_capture_enabled and not nvtx_capture_active and step == nvtx_capture_start_step:
            if accelerator.is_main_process:
                print(f"Starting NVTX capture range '{nvtx_capture_name}' at step {step}.")
            nvtx_push(nvtx_capture_name)
            nvtx_capture_active = True

    def maybe_stop_nvtx_capture(step):
        nonlocal nvtx_capture_active
        if nvtx_capture_enabled and nvtx_capture_active and step + 1 >= nvtx_capture_stop_step:
            if accelerator.is_main_process:
                print(f"Stopping NVTX capture range '{nvtx_capture_name}' after step {step}.")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            nvtx_pop()
            nvtx_capture_active = False
    
    # trainable_named_parameters = model.trainable_named_parameters()
    # # split the trainable parameters into 2 groups: name with pose_controller and w/o
    # pose_controller_named_parameters = [param for name, param in trainable_named_parameters if "pose_controller" in name]
    # other_named_parameters = [param for name, param in trainable_named_parameters if "pose_controller" not in name]
    # optimizer = torch.optim.AdamW([
    #     {'params': pose_controller_named_parameters, 'lr': learning_rate * 10.},
    #     {'params': other_named_parameters}
    # ], lr=learning_rate, weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    base_model = accelerator.unwrap_model(model)
    if online_unit_prefetch and dataset.load_from_cache:
        online_unit_prefetch = False
        if accelerator.is_main_process:
            print("online_unit_prefetch is disabled because dataset is loaded from cache.")
    if online_unit_prefetch and getattr(base_model.pipe, "vram_management_enabled", False):
        online_unit_prefetch = False
        if accelerator.is_main_process:
            print("online_unit_prefetch is disabled because pipeline VRAM management/offload is enabled.")

    def train_one_prepared_batch(inputs, pbar, step):
        maybe_start_nvtx_capture(step)
        maybe_start_cuda_profiler(step)
        nvtx_push("train_step/prepared_inputs")
        with accelerator.accumulate(model):
            try:
                optimizer.zero_grad()
                loss = model({}, inputs=inputs, inputs_prepared=True)
                if release_cache_before_backward:
                    # Return unused blocks to CUDA before NCCL needs device memory.
                    torch.cuda.empty_cache()
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, loss, save_steps)
                scheduler.step()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
            finally:
                nvtx_pop()
                maybe_stop_cuda_profiler(step)
                maybe_stop_nvtx_capture(step)

    def train_one_batch(data, pbar, step):
        maybe_start_nvtx_capture(step)
        maybe_start_cuda_profiler(step)
        nvtx_push("train_step/raw_data")
        with accelerator.accumulate(model):
            try:
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                if release_cache_before_backward:
                    # Live model tensors stay on the device; this is not offload.
                    torch.cuda.empty_cache()
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, loss, save_steps)
                scheduler.step()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
            finally:
                nvtx_pop()
                maybe_stop_cuda_profiler(step)
                maybe_stop_nvtx_capture(step)

    # losses = []
    
    if start_epoch >= num_epochs:
        print(f"Resume start epoch {start_epoch} is >= num_epochs {num_epochs}; no epochs left to train.")

    global_step = start_step
    for epoch_id in range(start_epoch, num_epochs):
        if online_unit_prefetch:
            pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch_id+1}/{num_epochs}', leave=True)
            prefetcher = OnlineUnitPrefetcher(
                base_model,
                iter(dataloader),
                queue_size=online_unit_prefetch_queue_size,
            )
            try:
                while True:
                    inputs = prefetcher.result()
                    if inputs is None:
                        break
                    train_one_prepared_batch(inputs, pbar, global_step)
                    global_step += 1
                    pbar.update(1)
            finally:
                prefetcher.close()
                pbar.close()
        else:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch_id+1}/{num_epochs}', leave=True)
            for data in pbar:
                train_one_batch(data, pbar, global_step)
                global_step += 1

                # losses.append(loss.item())
                # torch.save(losses, os.path.join(model_logger.output_path, 'losses.pth'))
                
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    if cuda_profiler_active:
        if accelerator.is_main_process:
            print("Stopping CUDA profiler at training end.")
        cuda_profiler_stop()
    if nvtx_capture_active:
        if accelerator.is_main_process:
            print(f"Stopping NVTX capture range '{nvtx_capture_name}' at training end.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        nvtx_pop()
    model_logger.on_training_end(accelerator, model, save_steps)

    accelerator.end_training()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
