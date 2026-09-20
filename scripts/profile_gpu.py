"""Systems profiling: hardware, CUDA smoke, single-job, torch.compile and concurrency benchmarks."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from src.config import Config
from src.data.dataset import prepare_data
from src.models import build_model
from src.training.checkpoint import write_json
from src.training.trainer import (autocast, device_and_precision, loss_value, make_loader,
                                  seed_everything)

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ("ofi_lstm", "hfformer", "patchtst", "moderntcn", "lit")
HISTORY_ROWS, BATCH_SIZE = 49, 128
SAFETY_FREE_BYTES = 3*1024**3      # Section R: at least 3 GiB free while jobs share the GPU.
COMPILE_SPEEDUP = 1.10             # Section S: compile is only worth it if throughput materially improves.
# Frozen architecture contract; a mismatch means the benchmark is not measuring the frozen model.
EXPECTED_PARAMETERS = dict(e0=0, ofi_lstm=55491, hfformer=22026, patchtst=477059,
                           moderntcn=50568195, lit=736547)
EXPECTED_CHANNELS = dict(e0=40, ofi_lstm=20, hfformer=38, patchtst=40, moderntcn=40, lit=40)
CONFIG_PATHS = dict(e0="configs/e0_60s.json") | {m: f"configs/{m}_60s_base.json" for m in LEARNED}
# (warmup steps, measured-step floor, measured-window floor in seconds); the window floor keeps
# NVML utilization meaningful and keeps concurrent jobs overlapping instead of finishing in turn.
DEFAULT_STEPS = dict(single=(20, 80, 3.), compile=(10, 40, 2.), worker=(10, 60, 8.),
                     concurrency=(10, 60, 8.))
EPOCH_SAMPLES = 58774  # Frozen train-split window count; the workload one model must finish per epoch.
# Section R: ModernTCN alone first, every model alone as the sequential baseline, then 2 -> 5 jobs.
# The ModernTCN pairings are the "ModernTCN with one or more light models" case section R allows.
CONCURRENCY_GROUPS = ([["moderntcn"]]+[[m] for m in LEARNED if m != "moderntcn"]
                      + [["ofi_lstm", "hfformer"],
                         ["moderntcn", "ofi_lstm"],
                         ["ofi_lstm", "hfformer", "lit"],
                         ["moderntcn", "ofi_lstm", "hfformer"],
                         ["ofi_lstm", "hfformer", "lit", "patchtst"],
                         ["ofi_lstm", "hfformer", "lit", "patchtst", "moderntcn"]])


def experiment(model, **overrides):
    config = Config.load(ROOT/CONFIG_PATHS[model])
    # e0/PatchTST/ModernTCN share the 40-channel raw-book branch, so a mislabelled config would
    # otherwise benchmark one architecture on another's features without tripping a shape check.
    if config.model != model:
        raise ValueError(f"{CONFIG_PATHS[model]} is a {config.model} run, not {model}")
    config.data.csv_path = str(ROOT/config.data.csv_path)  # Profiling must not depend on cwd.
    for key, value in overrides.items():
        setattr(config.training, key, value)
    config.validate()
    if config.training.batch_size != BATCH_SIZE:
        raise ValueError("Batch size is frozen at 128 for every profiling job")
    return config


def precision_name(dtype):
    return "fp32" if dtype is None else str(dtype).split(".")[-1]


def gib(value):
    return value/1024**3


class GpuSampler(threading.Thread):
    """Device-wide NVML sampling; concurrent jobs deliberately share one utilization number."""
    def __init__(self, index=0, interval=.1):
        super().__init__(daemon=True)
        self.interval, self.samples, self.stopped = interval, [], threading.Event()
        self.index, self.nvml, self.handle = index, None, None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml, self.handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            self.nvml = None  # nvidia-smi fallback below.

    def read(self):
        if self.nvml is not None:
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            return time.time(), float(util), int(memory.used), int(memory.free), int(memory.total)
        query = ["nvidia-smi", f"--id={self.index}", "--format=csv,noheader,nounits",
                 "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total"]
        fields = subprocess.run(query, capture_output=True, text=True, check=True).stdout.split(",")
        util, used, free, total = (float(f) for f in fields)
        return time.time(), util, int(used*1024**2), int(free*1024**2), int(total*1024**2)

    def run(self):
        while True:
            try:
                self.samples.append(self.read())
            except Exception:
                pass
            if self.stopped.wait(self.interval):
                return

    def stop(self):
        self.stopped.set()
        self.join(timeout=10)

    def window(self, start=None, end=None):
        rows = [s for s in self.samples
                if (start is None or s[0] >= start) and (end is None or s[0] <= end)] or self.samples
        if not rows:
            return dict(gpu_samples=0)
        util = [r[1] for r in rows]
        return dict(gpu_samples=len(rows), average_gpu_utilization_percent=sum(util)/len(util),
                    max_gpu_utilization_percent=max(util),
                    peak_memory_used_bytes=max(r[2] for r in rows),
                    min_free_bytes=min(r[3] for r in rows), memory_total_bytes=rows[0][4])


def nvml_versions():
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        return driver, f"{cuda//1000}.{cuda%1000//10}"
    except Exception:
        return None, None


def synthetic_batch(name, batch, generator):
    """Shape and scale the probe input per model; HFformer sees RAW prices, others standardized."""
    shape = ((batch, HISTORY_ROWS, 2, 10, 2) if name == "lit"
             else (batch, HISTORY_ROWS, EXPECTED_CHANNELS[name]))
    x = torch.randn(shape, generator=generator)
    if name == "hfformer":
        x = x.mul(5.)
        x[..., 0:9] += 27000.      # bid prices
        x[..., 18:27] += 27000.    # ask prices
        x[..., 36] *= 4e-6         # lagged log return
        x[..., 37] += 27000.       # weighted mid price
        x[..., 9:18] = x[..., 9:18].abs()/5.+.1    # bid sizes
        x[..., 27:36] = x[..., 27:36].abs()/5.+.1  # ask sizes
    return x, torch.randn(batch, 3, generator=generator).mul(1e-4)


def release(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def batches(loader, batch_size):
    """Full batches only: the short epoch-tail batch would trigger a torch.compile recompilation
    in the middle of a measured window and make compiled throughput look far worse than it is."""
    while True:
        for batch in loader:
            if len(batch[0]) == batch_size:
                yield batch


def train_step(forward, optimizer, batch, device, dtype, train_config):
    x, y = batch
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    with autocast(device, dtype):
        pred = forward(x)
        loss = loss_value(pred, y, train_config)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # Keep the finite check on device: a per-step .item() would serialize the pipeline being timed.
    return torch.isfinite(loss) & torch.isfinite(pred).all()


def run_benchmark(name, config, warmup_steps, measured_steps, min_seconds=0., compiled=False, barrier=None):
    """Temporary model instance, real train split, never a checkpoint and never fit()."""
    train_config = config.training
    device, dtype = device_and_precision(train_config)
    if device.type != "cuda":
        raise RuntimeError("Profiling requires CUDA")
    seed_everything(train_config.seed)
    data = prepare_data(config)
    if data.history_rows != HISTORY_ROWS or data.channels != EXPECTED_CHANNELS[name]:
        raise AssertionError(f"{name}: frozen shape is {HISTORY_ROWS}x{EXPECTED_CHANNELS[name]}, "
                             f"got {data.history_rows}x{data.channels}")
    # EPOCH_SAMPLES drives the schedule estimate, so measure it rather than trust it.
    if len(data.datasets["train"]) != EPOCH_SAMPLES:
        raise AssertionError(f"{name}: train split has {len(data.datasets['train'])} samples, "
                             f"frozen count is {EPOCH_SAMPLES}")
    model = build_model(name, data.history_rows, data.channels, **config.model_kwargs).to(device).train()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS[name]:
        raise AssertionError(f"{name}: {parameters} parameters, frozen count is {EXPECTED_PARAMETERS[name]}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                  weight_decay=train_config.weight_decay)
    loader = make_loader(data.datasets["train"], train_config, device, True,
                         torch.Generator().manual_seed(train_config.seed))
    stream = batches(loader, train_config.batch_size)
    release(device)
    forward, compile_seconds = model, None
    if compiled:
        started = time.time()
        forward = torch.compile(model)
        train_step(forward, optimizer, next(stream), device, dtype, train_config)
        torch.cuda.synchronize(device)
        compile_seconds = time.time()-started
    for _ in range(warmup_steps):
        train_step(forward, optimizer, next(stream), device, dtype, train_config)
    torch.cuda.synchronize(device)
    barrier_complete = barrier() if barrier is not None else None
    sampler = GpuSampler()
    sampler.start()
    finite = torch.ones((), dtype=torch.bool, device=device)
    started, steps = time.time(), 0
    while True:
        finite &= train_step(forward, optimizer, next(stream), device, dtype, train_config)
        steps += 1
        # Synchronize periodically: an unbounded async queue would make the wall clock fiction.
        if steps % 64 == 0 or steps >= measured_steps:
            torch.cuda.synchronize(device)
            if steps >= measured_steps and time.time()-started >= min_seconds:
                break
    ended = time.time()
    sampler.stop()
    seconds = ended-started
    samples = steps*train_config.batch_size
    result = dict(model=name, parameters=parameters, batch_size=train_config.batch_size,
                  num_workers=train_config.num_workers, prefetch_factor=train_config.prefetch_factor,
                  warmup_steps=warmup_steps, measured_steps=steps,
                  measured_steps_floor=measured_steps, measured_seconds_floor=min_seconds,
                  precision=precision_name(dtype), compiled=bool(compiled),
                  compile_seconds=compile_seconds, finite=bool(finite),
                  mean_step_seconds=seconds/steps, samples_per_second=samples/seconds,
                  measured_seconds=seconds, measured_start=started, measured_end=ended,
                  barrier_complete=barrier_complete,
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device))) | sampler.window()
    del model, forward, optimizer, stream, loader, data, finite
    release(device)
    return result


def mode_hardware(args):
    # device_and_precision also applies the frozen TF32 policy, so the flags below are the real ones.
    device, dtype = device_and_precision(Config().training)
    free, total = torch.cuda.mem_get_info(device)
    properties = torch.cuda.get_device_properties(device)
    driver, cuda_driver = nvml_versions()
    report = dict(hostname=platform.node(), platform=platform.platform(),
                  python_version=platform.python_version(), torch_version=torch.__version__,
                  cuda_runtime_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                  driver_version=driver, cuda_driver_version=cuda_driver,
                  gpu_name=properties.name, gpu_count=torch.cuda.device_count(),
                  compute_capability=f"{properties.major}.{properties.minor}",
                  multi_processor_count=properties.multi_processor_count,
                  vram_total_bytes=int(total), vram_free_bytes=int(free),
                  bf16_supported=bool(torch.cuda.is_bf16_supported()),
                  selected_precision=precision_name(dtype),
                  tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                  tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                  cpu_count=os.cpu_count(),
                  ram_total_bytes=os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))
    write_json(args.output_dir/"hardware.json", report)
    print(f"hardware | {report['gpu_name']} sm{report['compute_capability'].replace('.', '')} "
          f"{gib(total):.1f} GiB ({gib(free):.1f} free) | torch {torch.__version__} "
          f"cuda {torch.version.cuda} driver {driver} | bf16={report['bf16_supported']} "
          f"tf32={report['tf32_matmul']} | {report['cpu_count']} cpu "
          f"{gib(report['ram_total_bytes']):.0f} GiB ram", flush=True)
    return report


def mode_cuda_smoke(args):
    """One forward + backward per frozen architecture on synthetic data; no dataset, no checkpoint."""
    device, dtype = device_and_precision(Config().training)
    report = dict(batch_size=args.smoke_batch, history_rows=HISTORY_ROWS,
                  precision=precision_name(dtype), device=torch.cuda.get_device_name(device),
                  passed=True, models={})
    for name in ("e0",)+LEARNED:
        entry = dict(expected_parameters=EXPECTED_PARAMETERS[name], passed=False, error=None)
        try:
            seed_everything(42)
            generator = torch.Generator().manual_seed(42)
            model = build_model(name, HISTORY_ROWS, EXPECTED_CHANNELS[name]).to(device).train()
            parameters = sum(p.numel() for p in model.parameters())
            trainable = [p for p in model.parameters() if p.requires_grad]
            entry.update(parameters=parameters, trainable_tensors=len(trainable))
            if parameters != EXPECTED_PARAMETERS[name]:
                raise AssertionError(f"{name}: {parameters} parameters, frozen count is "
                                     f"{EXPECTED_PARAMETERS[name]}")
            x, y = synthetic_batch(name, args.smoke_batch, generator)
            x, y = x.to(device), y.to(device)
            release(device)
            started = time.time()
            with autocast(device, dtype):
                pred = model(x)
                loss = F.mse_loss(pred.float(), y.float())
            if tuple(pred.shape) != (args.smoke_batch, 3):
                raise AssertionError(f"{name}: output {tuple(pred.shape)}, expected [B,3]")
            if not torch.isfinite(pred).all():
                raise AssertionError(f"{name}: nonfinite predictions")
            if not torch.isfinite(loss):
                raise AssertionError(f"{name}: nonfinite loss")
            gradients = 0
            if trainable:
                loss.backward()
                grads = [p.grad for p in trainable if p.grad is not None]
                if len(grads) != len(trainable):
                    raise AssertionError(f"{name}: {len(trainable)-len(grads)} parameters without gradient")
                if not all(torch.isfinite(g).all() for g in grads):
                    raise AssertionError(f"{name}: nonfinite gradients")
                gradients = len(grads)
            torch.cuda.synchronize(device)
            entry.update(input_shape=list(x.shape), output_shape=list(pred.shape), finite=True,
                         loss=float(loss.detach()), backward=bool(trainable), gradient_tensors=gradients,
                         peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                         seconds=time.time()-started, passed=True)
            del model, x, y, pred, loss
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"[:800]
            report["passed"] = False
        release(device)
        report["models"][name] = entry
        print(f"cuda-smoke | {name:9s} {'PASS' if entry['passed'] else 'FAIL'} "
              f"params={entry.get('parameters')} shape={entry.get('input_shape')} "
              f"-> {entry.get('output_shape')} {entry['error'] or ''}", flush=True)
    write_json(args.output_dir/"cuda_smoke.json", report)
    if not report["passed"]:
        raise SystemExit("CUDA smoke tests failed; see reports/vast/cuda_smoke.json")
    return report


def mode_single(args):
    warmup, steps, seconds = steps_for(args, "single")
    report = dict(batch_size=BATCH_SIZE, num_workers=args.num_workers,
                  prefetch_factor=args.prefetch_factor, warmup_steps=warmup,
                  measured_steps_floor=steps, measured_seconds_floor=seconds,
                  device=torch.cuda.get_device_name(0), models={})
    for name in args.models:
        config = experiment(name, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
        result = run_benchmark(name, config, warmup, steps, seconds)
        report["models"][name] = result
        print(f"single | {name:9s} {result['samples_per_second']:8.1f} samples/s "
              f"step={result['mean_step_seconds']*1e3:6.2f} ms "
              f"alloc={gib(result['peak_allocated_bytes']):.2f} GiB "
              f"reserved={gib(result['peak_reserved_bytes']):.2f} GiB "
              f"util={result.get('average_gpu_utilization_percent', 0):.1f}%", flush=True)
    write_json(args.output_dir/"single_job_benchmark.json", report)
    return report


def mode_compile(args):
    warmup, steps, seconds = steps_for(args, "compile")
    report = dict(batch_size=BATCH_SIZE, warmup_steps=warmup, measured_steps_floor=steps,
                  measured_seconds_floor=seconds, speedup_threshold=COMPILE_SPEEDUP,
                  note="compile_seconds times the first compiled step in this process; a warm "
                       "inductor cache makes it far smaller than a cold first build. Windows "
                       "measure full 128-row batches only, so the steady state is timed without "
                       "the short epoch-tail batch that forces one dynamic-shape recompile.",
                  models={})
    for name in args.models:
        config = experiment(name, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
        eager = run_benchmark(name, config, warmup, steps, seconds)
        entry = dict(eager=eager, compiled=None, compile_succeeded=False, error=None)
        try:
            entry["compiled"] = run_benchmark(name, config, warmup, steps, seconds, compiled=True)
            entry["compile_succeeded"] = True
        except Exception as error:  # A compile failure must not end the sweep (section S).
            entry["error"] = f"{type(error).__name__}: {error}"[:800]
            release(torch.device("cuda"))
        try:
            torch.compiler.reset()
        except Exception:
            pass
        compiled = entry["compiled"]
        speedup = compiled["samples_per_second"]/eager["samples_per_second"] if compiled else None
        entry["speedup"] = speedup
        entry["compile_seconds"] = compiled["compile_seconds"] if compiled else None
        entry["finite"] = bool(compiled["finite"]) if compiled else False
        # Section S: enable only when compile is reliable, finite and materially faster.
        entry["enable"] = bool(compiled and compiled["finite"] and speedup >= COMPILE_SPEEDUP)
        report["models"][name] = entry
        print(f"compile | {name:9s} eager={eager['samples_per_second']:8.1f} "
              f"compiled={compiled['samples_per_second'] if compiled else float('nan'):8.1f} samples/s "
              f"speedup={speedup if speedup else float('nan'):.2f}x "
              f"build={entry['compile_seconds'] or float('nan'):.1f}s "
              f"enable={entry['enable']} {entry['error'] or ''}", flush=True)
    write_json(args.output_dir/"compile_benchmark.json", report)
    return report


def wait_for_peers(directory, tag, group_size, timeout=180.):
    """Overlap the measured windows: nobody times anything until every peer finished warmup."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f"{tag}.ready").write_text(str(os.getpid()))
    deadline = time.time()+timeout  # A peer killed outside Python leaves no marker; do not hang on it.
    while time.time() < deadline:
        if len(list(directory.glob("*.ready"))) >= group_size:
            return True
        if any(directory.glob("*.failed")):  # A dead peer must not hang the survivors.
            return False
        time.sleep(.05)
    return False


def mode_worker(args):
    """One benchmark process; the concurrency parent aggregates these JSON results."""
    warmup, steps, seconds = steps_for(args, "worker")
    result = dict(model=args.model, pid=os.getpid(), oom=False, error=None,
                  group_size=args.group_size)
    barrier = None
    if args.barrier_dir:
        barrier = lambda: wait_for_peers(args.barrier_dir, f"{args.model}_{os.getpid()}", args.group_size)
    try:
        config = experiment(args.model, num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor)
        result |= run_benchmark(args.model, config, warmup, steps, seconds, barrier=barrier)
    except torch.cuda.OutOfMemoryError as error:
        result |= dict(oom=True, error=f"CUDA OOM: {error}"[:800])
    except Exception as error:
        result |= dict(error=f"{type(error).__name__}: {error}"[:800])
    if result["error"] and args.barrier_dir:
        Path(args.barrier_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.barrier_dir)/f"{args.model}_{os.getpid()}.failed").write_text(result["error"])
    write_json(args.result_path, result)
    print(f"worker | {args.model} pid={result['pid']} "
          f"{result.get('samples_per_second', float('nan')):.1f} samples/s "
          f"{result['error'] or ''}", flush=True)
    return result


def run_group(models, args, temp_root, baseline, solo_util=None):
    warmup, steps, seconds = steps_for(args, "concurrency")
    directory = Path(tempfile.mkdtemp(dir=temp_root))
    environment = os.environ | dict(CUDA_VISIBLE_DEVICES="0")
    processes = []
    for index, name in enumerate(models):
        path = directory/f"{index}_{name}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "worker",
                   "--model", name, "--steps", str(steps), "--warmup", str(warmup),
                   "--min-seconds", str(seconds),
                   "--num-workers", str(args.num_workers),
                   "--prefetch-factor", str(args.prefetch_factor), "--result-path", str(path),
                   "--barrier-dir", str(directory/"barrier"), "--group-size", str(len(models))]
        processes.append((name, path, subprocess.Popen(command, env=environment, text=True,
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.STDOUT)))
    sampler = GpuSampler()
    sampler.start()
    runs, failures = [], []
    try:
        for name, path, process in processes:
            try:
                output = process.communicate(timeout=args.group_timeout)[0]
            except subprocess.TimeoutExpired:
                process.kill()
                output = process.communicate()[0]
                failures.append(f"{name}: timed out after {args.group_timeout:.0f}s")
            try:  # A worker killed mid-write leaves truncated JSON; that is a failure, not a crash.
                run = json.loads(path.read_text())
            except (OSError, ValueError):
                failures.append(f"{name}: no usable result (exit {process.returncode}): "
                                f"{output[-400:].strip()}")
                continue
            runs.append(run)
            if run.get("error"):
                failures.append(f"{name}: {run['error']}")
    finally:
        sampler.stop()
        for _, _, process in processes:  # Never leave a worker holding VRAM behind.
            if process.poll() is None:
                process.kill()
                process.communicate()
    complete = [r for r in runs if not r.get("error") and r.get("measured_seconds")]
    usage = (sampler.window(min(r["measured_start"] for r in complete),
                            max(r["measured_end"] for r in complete)) if complete else sampler.window())
    group = dict(models=list(models), size=len(models), warmup_steps=warmup,
                 measured_steps_floor=steps, measured_seconds_floor=seconds,
                 processes=[{k: r.get(k) for k in
                             ("model", "pid", "samples_per_second", "mean_step_seconds",
                              "peak_allocated_bytes", "peak_reserved_bytes", "measured_seconds",
                              "measured_start", "measured_end",  # Keeps window overlap auditable.
                              "finite", "barrier_complete", "oom", "error")} for r in runs],
                 oom=any(r.get("oom") for r in runs), failures=failures,
                 peak_memory_used_bytes=usage.get("peak_memory_used_bytes"),
                 min_free_bytes=usage.get("min_free_bytes"),
                 average_gpu_utilization_percent=usage.get("average_gpu_utilization_percent"),
                 max_gpu_utilization_percent=usage.get("max_gpu_utilization_percent"),
                 gpu_samples=usage.get("gpu_samples", 0))
    samples = sum(r["measured_steps"]*r["batch_size"] for r in complete)
    concurrent_seconds = (max(r["measured_end"] for r in complete)
                          - min(r["measured_start"] for r in complete)) if complete else None
    # Sequential baseline: the same work run one job after another at single-job throughput.
    rates = {r["model"]: baseline.get(r["model"]) or (r["samples_per_second"] if len(models) == 1 else None)
             for r in complete}
    sequential_seconds = (sum(r["measured_steps"]*r["batch_size"]/rates[r["model"]] for r in complete)
                          if complete and all(rates.values()) else None)
    group |= dict(total_samples=samples, concurrent_seconds=concurrent_seconds,
                  sequential_seconds=sequential_seconds,
                  aggregate_samples_per_second=samples/concurrent_seconds if concurrent_seconds else None,
                  sequential_samples_per_second=samples/sequential_seconds if sequential_seconds else None,
                  speedup_vs_sequential=(sequential_seconds/concurrent_seconds
                                         if sequential_seconds and concurrent_seconds else None))
    reasons = []
    if failures or len(complete) != len(models):
        reasons.append("job failure" if not group["oom"] else "CUDA OOM")
    if not all(r["finite"] for r in complete):
        reasons.append("nonfinite loss/predictions")
    if not group["gpu_samples"]:  # Unsampled means the VRAM margin below was never actually checked.
        reasons.append("no GPU telemetry for the measured window")
    elif group["min_free_bytes"] < SAFETY_FREE_BYTES:
        reasons.append(f"free VRAM {gib(group['min_free_bytes']):.2f} GiB below the 3 GiB margin")
    if any(r.get("barrier_complete") is False for r in complete):
        reasons.append("measured windows did not overlap")
    # Section R criterion 2: sharing the GPU has to raise device utilization, not only aggregate rate.
    solo = [u for u in ((solo_util or {}).get(m) for m in models) if u]
    if len(models) > 1 and solo and group["gpu_samples"] \
            and group["average_gpu_utilization_percent"] <= max(solo):
        reasons.append(f"GPU utilization {group['average_gpu_utilization_percent']:.1f}% not above "
                       f"the {max(solo):.1f}% single-job baseline")
    if group["speedup_vs_sequential"] is None:
        reasons.append("no sequential baseline")
    elif len(models) > 1 and group["speedup_vs_sequential"] <= 1.:
        reasons.append("aggregate throughput not above sequential")
    group |= dict(accepted=not reasons, rejection_reasons=reasons)
    return group


def recommend(groups, workload):
    """Partition the five models over the measured groups, minimizing total wall time for an equal
    per-model workload. A concurrent group is only as fast as its slowest member, so the fastest
    aggregate throughput is not automatically the fastest schedule."""
    options = {}
    for group in groups:
        rates = [p["samples_per_second"] for p in group["processes"] if p.get("samples_per_second")]
        if group["accepted"] and len(rates) == len(group["models"]):
            options[frozenset(group["models"])] = workload/min(rates)
    best = None

    def search(remaining, chosen, cost):
        nonlocal best
        if best is not None and cost >= best[1]:
            return
        if not remaining:
            best = (chosen, cost)
            return
        anchor = sorted(remaining)[0]  # Fix one model per level: enumerate partitions, not orders.
        for models, seconds in options.items():
            if anchor in models and models <= remaining:
                search(remaining-models, chosen+[sorted(models)], cost+seconds)

    search(frozenset(LEARNED), [], 0.)
    alone = [options.get(frozenset([m])) for m in LEARNED]
    sequential = sum(alone) if all(alone) else None  # A rejected solo run leaves no baseline.
    if best is None:  # Nothing measurable covers every model; fall back to one job at a time.
        return [[m] for m in LEARNED], None, sequential
    return best[0], best[1], sequential


def mode_concurrency(args):
    single_path = args.output_dir/"single_job_benchmark.json"
    baseline, solo_util, source = {}, {}, "measured single-job groups in this run"
    if single_path.exists():
        previous = json.loads(single_path.read_text())["models"]
        baseline = {name: entry["samples_per_second"] for name, entry in previous.items()}
        solo_util = {name: entry.get("average_gpu_utilization_percent")
                     for name, entry in previous.items()}
        source = f"{single_path} plus single-job groups measured here"
    measurements = []
    with tempfile.TemporaryDirectory(prefix="profile-concurrency-") as temp:
        for models in CONCURRENCY_GROUPS:
            if any(m not in args.models for m in models):
                continue
            group = run_group(models, args, Path(temp), baseline, solo_util)
            if len(models) == 1 and group["accepted"]:
                baseline[models[0]] = group["aggregate_samples_per_second"]
                # Same sampler and window length as the concurrent groups: the fair comparison.
                solo_util[models[0]] = group["average_gpu_utilization_percent"]
            measurements.append(group)
            print(f"concurrency | {'+'.join(models):45s} "
                  f"aggregate={group['aggregate_samples_per_second'] or float('nan'):8.1f} samples/s "
                  f"speedup={group['speedup_vs_sequential'] or float('nan'):.2f}x "
                  f"vram={gib(group['peak_memory_used_bytes'] or 0):5.2f} GiB "
                  f"free={gib(group['min_free_bytes'] or 0):5.2f} GiB "
                  f"util={group['average_gpu_utilization_percent'] or 0:5.1f}% "
                  f"{'ACCEPT' if group['accepted'] else 'REJECT '+'; '.join(group['rejection_reasons'])}",
                  flush=True)
    epochs = experiment(args.models[0]).training.epochs
    workload = EPOCH_SAMPLES*epochs
    schedule, wall, sequential_wall = recommend(measurements, workload)
    best = [g for g in measurements if g["accepted"] and g["size"] > 1]

    def summarize(group):
        slowest = min(p["samples_per_second"] for p in group["processes"])
        return (f"{'+'.join(group['models'])}: {group['speedup_vs_sequential']:.2f}x aggregate vs "
                f"sequential, slowest member {slowest:.0f} samples/s, "
                f"{group['average_gpu_utilization_percent']:.0f}% util, "
                f"{gib(group['min_free_bytes']):.1f} GiB free")

    rationale = ("; ".join(summarize(g)
                           for g in sorted(best, key=lambda g: -g["speedup_vs_sequential"]))
                 or "no concurrent group beat the sequential baseline within the 3 GiB safety margin")
    saving = f"{(1-wall/sequential_wall)*100:.1f}% below" if wall and sequential_wall else "unknown against"
    cost = f"{wall:.0f}s" if wall else "an unmeasured amount"
    baseline_cost = f"{sequential_wall:.0f}s" if sequential_wall else "an unmeasured"
    decision = dict(recommended_groups=schedule, baseline_source=source,
                    safety_margin_bytes=SAFETY_FREE_BYTES, epochs=epochs,
                    workload_samples_per_model=workload,
                    estimated_train_step_seconds=wall,
                    sequential_train_step_seconds=sequential_wall,
                    accepted_groups=[g["models"] for g in measurements if g["accepted"]],
                    rejected_groups=[dict(models=g["models"], reasons=g["rejection_reasons"])
                                     for g in measurements if not g["accepted"]],
                    rationale=f"Measured, not assumed. {rationale}. Schedules are ranked by the wall "
                              f"time to push {workload} train samples through every model, each group "
                              f"paced by its slowest member: {' then '.join('+'.join(g) for g in schedule)} "
                              f"costs {cost} of train steps, {saving} the {baseline_cost} "
                              f"one-job-at-a-time baseline (forward/backward/AdamW only; validation, "
                              f"checkpointing and data-loader startup excluded).")
    report = dict(batch_size=BATCH_SIZE, num_workers=args.num_workers,
                  prefetch_factor=args.prefetch_factor,
                  parent_cuda_initialized=torch.cuda.is_initialized(),
                  groups=measurements, decision=decision)
    write_json(args.output_dir/"concurrency_benchmark.json", report)
    print(f"concurrency | decision: {decision['rationale']}", flush=True)
    return report


def steps_for(args, mode):
    warmup, steps, seconds = DEFAULT_STEPS[mode]
    return (args.warmup if args.warmup is not None else warmup,
            args.steps if args.steps is not None else steps,
            args.min_seconds if args.min_seconds is not None else seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=("hardware", "cuda-smoke", "single", "compile", "worker", "concurrency", "all"))
    p.add_argument("--models", default=",".join(LEARNED), help="Comma list for single/compile/concurrency")
    p.add_argument("--model", choices=LEARNED, help="Worker mode only")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--warmup", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--min-seconds", type=float, help="Measured-window floor, seconds")
    p.add_argument("--smoke-batch", type=int, default=8)
    p.add_argument("--result-path", help="Worker mode only")
    p.add_argument("--barrier-dir", help="Worker mode only")
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--group-timeout", type=float, default=900.)
    p.add_argument("--output-dir", default=str(ROOT/"reports"/"vast"))
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(args.models)-set(LEARNED)
    if not args.models:
        p.error("--models cannot be empty")
    if unknown:
        p.error(f"Benchmarks cover the five learned models only; unknown: {sorted(unknown)}")
    if args.mode == "worker" and not (args.model and args.result_path):
        p.error("Worker mode needs --model and --result-path")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    modes = dict(hardware=mode_hardware, cuda_smoke=mode_cuda_smoke, single=mode_single,
                 compile=mode_compile, worker=mode_worker, concurrency=mode_concurrency)
    if args.mode == "all":
        for name in ("hardware", "cuda_smoke", "single", "compile"):
            modes[name](args)
        # Concurrency runs in a fresh process so the workers are not measured against this one's
        # allocator cache and inductor state. Its idle CUDA context still holds VRAM, which only
        # makes the measured free-memory margin conservative.
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "concurrency",
                   "--models", ",".join(args.models), "--num-workers", str(args.num_workers),
                   "--prefetch-factor", str(args.prefetch_factor),
                   "--group-timeout", str(args.group_timeout), "--output-dir", str(args.output_dir)]
        for flag, value in (("--steps", args.steps), ("--warmup", args.warmup),
                            ("--min-seconds", args.min_seconds)):
            command += [flag, str(value)] if value is not None else []
        raise SystemExit(subprocess.run(command, check=False).returncode)
    modes[args.mode.replace("-", "_")](args)


if __name__ == "__main__":
    main()
