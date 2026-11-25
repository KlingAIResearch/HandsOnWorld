"""Bound student autograd lifetime to one temporal chunk, retaining the DMD loss.

Wan's generated-prefix KV is already detached in the original Self-Forcing
algorithm. Thus dL/dtheta is the sum of the per-exit VJPs, with no gradient
through earlier chunks or earlier denoising steps. Keep the sampled trajectory
and its KV, evaluate the original full-video loss, then replay each exit and
backward immediately. Do not use FSDP no_sync: each VJP reduces to local shards.
"""

import torch


def streaming_student_backward(model, data, iteration, progress=None, verify_replay=False):
    """Perform student backward only; caller clips and steps the optimizer once.

    All chunks participate. Floating-point gradient accumulation order changes,
    so mathematical equivalence does not imply bitwise BF16 gradient equality.
    This entry is restricted to the no-GAN Wan control trainer.
    """
    if iteration % model.config.student_update_freq:
        raise ValueError("Chunk replay is only for student updates")
    if model.config.gan_loss_weight_gen or not model.config.enable_gradient_in_rollout:
        raise ValueError("Chunk replay requires no GAN and enabled rollout gradients")
    real, condition, negative = model._prepare_training_data(data)
    model._setup_grad_requirements(iteration)
    noise, t_student, t, eps = model._generate_noise_and_time(real, iteration=iteration)
    records = []

    def report(event, **fields):
        if progress is not None:
            progress(event, **fields)

    try:
        with torch.no_grad():
            generated = model.rollout_with_gradient(
                noise,
                condition=condition,
                enable_gradient=False,
                start_gradient_frame=model.config.start_gradient_frame,
                exit_records=records,
            )
        report("stream_rollout_done", chunks=len(records))
        leaf = generated.detach().requires_grad_(True)
        losses, outputs = model._student_update_step(
            noise,
            t_student,
            t,
            eps,
            data,
            condition=condition,
            neg_condition=negative,
            gen_data_override=leaf,
        )
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError("Nonfinite streaming DMD loss")
        # Preserve original whole-video normalization and all CFG/score logic.
        (video_grad,) = torch.autograd.grad(losses["total_loss"], leaf)
        detached_losses = {name: value.detach() for name, value in losses.items()}
        del leaf, losses, outputs, noise, eps
        report("stream_score_done")
        device = generated.device
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=devices):
            # Reconstruct exactly the prefix cache visible at each recorded exit.
            # Replaying in reverse would expose every chunk to the final cache.
            model.net.clear_caches()
            if model.net.fixed_first_frame:
                prefix = generated[:, :, :1]
                with torch.no_grad():
                    model.net(
                        prefix,
                        torch.zeros(prefix.shape[0], device=prefix.device, dtype=prefix.dtype),
                        condition=condition,
                        cache_tag="pos",
                        store_kv=True,
                        cur_start_frame=0,
                        fwd_pred_type="x0",
                        is_ar=True,
                    )
            for record in records:
                start = record["start"]
                end = start + record["input"].shape[2]
                if start >= model.config.start_gradient_frame:
                    torch.set_rng_state(record["cpu_rng"])
                    if record["cuda_rng"] is not None:
                        torch.cuda.set_rng_state(record["cuda_rng"], device)
                    report("stream_chunk_forward", start=start, end=end)
                    with torch.enable_grad():
                        prediction = model.net(
                            record["input"],
                            record["t"],
                            condition=condition,
                            cache_tag="pos",
                            store_kv=False,
                            cur_start_frame=start,
                            fwd_pred_type="x0",
                            is_ar=True,
                        )
                    if verify_replay:
                        torch.testing.assert_close(
                            prediction.detach(), generated[:, :, start:end], rtol=1e-3, atol=1e-4
                        )
                    prediction.backward(video_grad[:, :, start:end])
                    del prediction
                    report("stream_chunk_backward_done", start=start, end=end)
                # Even chunks excluded from gradient replay form the causal
                # prefix required by later trainable chunks.
                with torch.no_grad():
                    model.net(
                        record["cache_input"],
                        record["cache_t"],
                        condition=condition,
                        cache_tag="pos",
                        store_kv=True,
                        cur_start_frame=start,
                        fwd_pred_type="x0",
                        is_ar=True,
                    )
        return detached_losses, {}
    finally:
        records.clear()
        model.net.clear_caches()
