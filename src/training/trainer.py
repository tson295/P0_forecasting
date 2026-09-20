from contextlib import nullcontext
import copy
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from .checkpoint import (capture_rng, restore_rng, save_training_checkpoint,
                         load_training_state, verify_checkpoint, write_json)
from src.utils.metrics import ReturnMetrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_and_precision(config):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if config.device == "auto" else config.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Supported trainer devices: cpu or cuda[:index]")
    dtype = None
    if device.type == "cuda":
        # torch.cuda memory queries reject an index-less device, so pin one now.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        precision = config.precision
        if precision == "auto":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unavailable; use auto or fp16")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    elif config.precision not in ("auto", "fp32"):
        raise ValueError("CPU smoke/training uses FP32; use precision=auto or fp32")
    return device, dtype


def autocast(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def make_loader(dataset, config, device, shuffle=False, generator=None):
    # Separate worker seed generation from sampling so persistent worker recreation
    # at epoch-boundary resume cannot alter the next epoch's permutation.
    worker_generator = torch.Generator().manual_seed(config.seed+123)
    options = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                   pin_memory=device.type == "cuda", drop_last=False, generator=worker_generator)
    if shuffle:
        options["sampler"] = RandomSampler(dataset, generator=generator)
    if config.num_workers:
        options.update(persistent_workers=True, prefetch_factor=config.prefetch_factor)
    return DataLoader(dataset, **options)


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False)+"\n")


def truncate_jsonl(path, first_dropped_epoch):
    """Resume rewrites history from the resumed epoch; earlier epochs are kept."""
    path = Path(path)
    if not path.exists():
        return
    kept = [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["epoch"] < first_dropped_epoch]
    path.write_text("".join(line+"\n" for line in kept), encoding="utf-8")


def loss_value(pred, target, config):
    if config.loss == "mse":
        return nn.functional.mse_loss(pred.float(), target.float())
    error = target.float()-pred.float()
    return torch.maximum(config.quantile*error, (config.quantile-1)*error).mean()


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.eval()
    metrics = ReturnMetrics(device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, dtype):
            pred = model(x)
        metrics.update(pred, y)
    result = metrics.compute()
    if not math.isfinite(result["mean_mse"]):
        raise FloatingPointError("Nonfinite evaluation predictions")
    return result


def probe_batch_size(model, sample, config, device, dtype):
    """Opt-in CUDA probe, one forward/backward/AdamW step per candidate, <=6 by default.

    Uses a clone and restores all RNG state. Accounts for optimizer allocations and
    external GPU usage; retains at least configured free VRAM. Never called by smoke.
    """
    if device.type != "cuda":
        raise ValueError("Batch probing requires CUDA")
    rng = capture_rng()
    best = None
    records = []
    try:
        batch = 128
        while batch <= config.probe_max_batch_size:
            candidate = optimizer = x = y = loss = scaler = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                candidate = copy.deepcopy(model).to(device).train()
                optimizer = torch.optim.AdamW((p for p in candidate.parameters() if p.requires_grad),
                                               lr=config.learning_rate, weight_decay=config.weight_decay)
                scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
                x = sample.unsqueeze(0).expand(batch, *sample.shape).contiguous().to(device)
                y = torch.zeros(batch, 3, device=device)
                with autocast(device, dtype):
                    loss = loss_value(candidate(x), y, config)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize(device)
                free, total = torch.cuda.mem_get_info(device)
                peak = torch.cuda.max_memory_allocated(device)
                # Reserved memory is already excluded by mem_get_info.
                safe = free >= config.vram_headroom_gb*1024**3
                records.append(dict(batch_size=batch, peak_allocated_bytes=peak,
                                    free_bytes=free, total_bytes=total, accepted=safe))
                if not safe:
                    break
                best = batch
            except torch.cuda.OutOfMemoryError:
                records.append(dict(batch_size=batch, accepted=False, reason="CUDA OOM"))
                break
            finally:
                del candidate, optimizer, x, y, loss, scaler
                gc.collect()
                torch.cuda.empty_cache()
            batch *= 2
    finally:
        restore_rng(rng)
    if best is None:
        raise RuntimeError("Batch 128 does not fit with headroom; set a smaller batch manually")
    return best, records


def smoke_forward(model, dataset, backward=False):
    """Exactly one CPU batch (<=2 examples), no optimizer and no epoch iteration."""
    x = torch.stack([dataset[i][0] for i in range(min(2, len(dataset)))])
    y = torch.stack([dataset[i][1] for i in range(min(2, len(dataset)))])
    model.cpu().train(backward)
    with torch.set_grad_enabled(backward):
        pred = model(x)
        if pred.shape != (len(x), 3) or not torch.isfinite(pred).all():
            raise AssertionError(f"Invalid model output {pred.shape}")
        if backward and any(p.requires_grad for p in model.parameters()):
            nn.functional.mse_loss(pred, y).backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise AssertionError("Missing/nonfinite gradients")
            model.zero_grad(set_to_none=True)
    return dict(input_shape=list(x.shape), output_shape=list(pred.shape), finite=True,
                backward=backward and pred.requires_grad,
                parameters=sum(p.numel() for p in model.parameters()))


def fit(model, data, experiment, resume=None):
    """Only called by an explicit non-smoke, non-prepare CLI invocation."""
    config = experiment.training
    device, dtype = device_and_precision(config)
    run = Path(config.checkpoint_root)/experiment.model/config.run_name
    if run.exists() and not resume:
        raise FileExistsError(f"Run exists: {run}; select a new run_name or --resume")
    if resume and config.auto_batch_size:
        raise ValueError("Disable batch probing when resuming an exact training run")
    run.mkdir(parents=True, exist_ok=True)
    if config.auto_batch_size:
        config.batch_size, records = probe_batch_size(model, data.datasets["train"][0][0], config, device, dtype)
        write_json(run/"batch_probe.json", records)
    model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = make_loader(data.datasets["train"], config, device, True, generator)
    # A separate generator keeps validation worker creation out of training RNG.
    val_generator = torch.Generator().manual_seed(config.seed+1)
    val_loader = make_loader(data.datasets["validation"], config, device, generator=val_generator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    start, global_step, best = 0, 0, math.inf
    if resume:
        state = load_training_state(resume, optimizer, scheduler, scaler, generator)
        start, global_step, best = state["epoch"]+1, state["global_step"], state["best_validation_mse"]
        if state["head_only"] != model.head_only:
            raise ValueError("Resume fine-tuning mode mismatch")
    if start >= config.epochs:
        raise ValueError("Checkpoint has already completed the configured epoch budget")
    forward_model = torch.compile(model) if config.compile else model
    metadata = data.metadata | {"experiment": experiment.to_dict()}
    history_path = run/"training_history.jsonl"
    truncate_jsonl(history_path, start)
    verification_sample = torch.stack([data.datasets["validation"][i][0]
                                       for i in range(min(8, len(data.datasets["validation"])))])
    started = time.time()
    epoch_seconds, best_epoch = [], None
    for epoch in range(start, config.epochs):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_metrics = ReturnMetrics(device)
        batches = len(train_loader)
        group_samples = 0
        for batch_index, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device, dtype):
                pred = forward_model(x)
                loss = loss_value(pred, y, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            # Sum sample means, then divide gradients by actual group sample count.
            scaler.scale(loss*len(x)).backward()
            group_samples += len(x)
            train_metrics.update(pred, y)
            if (batch_index+1) % config.gradient_accumulation == 0 or batch_index+1 == batches:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_samples)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
                global_step += 1
        validation = evaluate(forward_model, val_loader, device, dtype)
        learning_rate = float(scheduler.get_last_lr()[0])
        scheduler.step()
        # Selection criterion: validation mean MSE across the three horizons only.
        improved = validation["mean_mse"] < best
        best = min(best, validation["mean_mse"])
        if improved:
            best_epoch = epoch
        state = dict(epoch=epoch, global_step=global_step, best_validation_mse=best,
                     head_only=model.head_only, scaler=scaler.state_dict(), rng=capture_rng(generator))
        # Contract V: verify each folder against the weights it was just written from.
        verification = {}
        for name in (("best",) if improved else ())+("last",):
            save_training_checkpoint(model, run/name, metadata, optimizer, scheduler, state)
            verification[name] = verify_checkpoint(run/name, model, verification_sample, device)
        epoch_seconds.append(time.time()-epoch_started)
        result = dict(epoch=epoch, train=train_metrics.compute(), validation=validation,
                      learning_rate=learning_rate, best_validation_mean_mse=best,
                      is_best=bool(improved), global_step=global_step,
                      epoch_seconds=epoch_seconds[-1], checkpoint_verification=verification,
                      peak_allocated_bytes=(int(torch.cuda.max_memory_allocated(device))
                                            if device.type == "cuda" else None),
                      peak_reserved_bytes=(int(torch.cuda.max_memory_reserved(device))
                                           if device.type == "cuda" else None))
        write_json(run/f"epoch_{epoch:04d}.json", result)
        append_jsonl(history_path, result)
        print(json.dumps({k: result[k] for k in ("epoch", "learning_rate", "is_best", "epoch_seconds")}
                         | {"train_mean_mse": result["train"]["mean_mse"],
                            "validation_mean_mse": validation["mean_mse"]}), flush=True)
    # `last` still matches the in-memory weights; `best` was verified at its own epoch.
    final_verification = dict(last=verify_checkpoint(run/"last", model, verification_sample, device),
                              best=dict(path=str(run/"best"), exact=True,
                                        verified_at_epoch=best_epoch,
                                        note="verified against the weights it was saved from"))
    summary = dict(model=experiment.model, run_name=config.run_name, run_directory=str(run),
                   epochs_completed=config.epochs, first_epoch_this_process=start,
                   best_epoch=best_epoch, best_validation_mean_mse=best,
                   total_seconds=time.time()-started, epoch_seconds=epoch_seconds,
                   batch_size=config.batch_size, num_workers=config.num_workers,
                   prefetch_factor=config.prefetch_factor, compile=bool(config.compile),
                   precision=("fp32" if dtype is None else str(dtype).split(".")[-1]),
                   device=str(device), seed=config.seed, pid=os.getpid(), host=platform.node(),
                   gpu=(torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                   peak_allocated_bytes=(int(torch.cuda.max_memory_allocated(device))
                                         if device.type == "cuda" else None),
                   peak_reserved_bytes=(int(torch.cuda.max_memory_reserved(device))
                                        if device.type == "cuda" else None),
                   checkpoint_verification=final_verification)
    write_json(run/"training_run.json", summary)
    # No implicit test evaluation; test remains held out until explicitly requested.
    return run
