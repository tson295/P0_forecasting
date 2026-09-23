"""Single-job and concurrency throughput benchmark for the classification sweep.

  python scripts/cls_benchmark.py --csv ~/data/BTC_L10_gate_1y.csv --out benchmarks

Every worker is a real training process (fold 1, equal-width, multi-horizon, the exact
job recipe: batch 128, bf16, compile as in the sweep). Workers prepare data, compile and
warm up, then wait at a file barrier so that all workers of a configuration measure the
same window. The controller samples GPU utilization and memory (NVML), system CPU, RAM
and disk I/O (psutil) during that window.

Throughput of a mixed configuration is compared through the normalized rate
sum_i samples_per_second_i / single_job_samples_per_second(arch_i): the number of
isolated jobs' worth of progress per second. Running jobs serially scores 1.0.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHORT = dict(T="moderntcn", F="transformer", L="ofi_lstm")
CONFIGS = ["T", "F", "L",
           "TT", "FF", "LL", "TF", "TL", "FL",
           "TTT", "FFF", "LLL", "TFL",
           "TTTT", "FFFF", "LLLL", "TTFL", "TFFL", "TFLL",
           "TTFFLL", "FFFLLL"]


def worker(args):
    import psutil
    import torch
    from torch import nn
    from src.cls.data import ClassificationData
    from src.cls.jobs import TRAINING, COMPILE, wf3_data_fields
    from src.cls.labels import LabelSet
    from src.cls.models import build_classifier, parameter_count
    from src.cls.trainer import training_functions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", 0)
    label_set = LabelSet.load(ROOT/"labeling/equal_width_k32_p99.json")
    t0 = time.time()
    data = ClassificationData(args.arch, wf3_data_fields(1), args.csv, label_set, [60, 120, 180], device)
    prepare_seconds = time.time()-t0
    torch.manual_seed(42)
    model = build_classifier(args.arch, data.history_rows, data.channels, data.n_classes(), [60, 120, 180]).to(device)
    compile_mode = COMPILE[args.arch]
    loss_fn, _ = training_functions(model, compile_mode, torch.bfloat16, device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"],
                            fused=True)
    batch = TRAINING["batch_size"]
    order = torch.randperm(len(data.splits["train"]), generator=torch.Generator().manual_seed(42))
    stream = data.batches("train", batch, order)

    def step():
        """Exactly the trainer's step (src/cls/trainer.fit)."""
        nonlocal stream
        # Full batches only: the epoch's final partial batch (333177 mod 128 = 121 samples)
        # would trigger a one-off recompile inside the measured window.
        x, y, _, _ = next(stream, (None, None, None, None))
        if x is None or len(y) < batch:
            stream = data.batches("train", batch, order)
            x, y, _, _ = next(stream)
        loss = loss_fn(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAINING["gradient_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        return len(y)

    t0 = time.time()
    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()
    warmup_seconds = time.time()-t0
    torch.cuda.reset_peak_memory_stats()
    Path(args.ready).touch()
    while not Path(args.go).exists():
        time.sleep(0.02)
    proc = psutil.Process()
    cpu0 = proc.cpu_times()
    t0 = time.time()
    samples, steps = 0, 0
    while time.time()-t0 < args.seconds:
        samples += step()
        steps += 1
    torch.cuda.synchronize()
    wall = time.time()-t0
    cpu1 = proc.cpu_times()
    result = dict(arch=args.arch, compile_mode=compile_mode, parameters=parameter_count(model), batch_size=batch,
                  samples=samples, steps=steps, seconds=wall, samples_per_second=samples/wall,
                  mean_step_seconds=wall/steps,
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  resident_feature_bytes=int(data.x.numel()*data.x.element_size()),
                  process_cpu_percent=100*((cpu1.user-cpu0.user)+(cpu1.system-cpu0.system))/wall,
                  rss_bytes=int(proc.memory_info().rss), prepare_seconds=prepare_seconds,
                  warmup_seconds=warmup_seconds, warmup_steps=args.warmup_steps)
    Path(args.out).write_text(json.dumps(result, indent=2)+"\n")


class Sampler(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        import psutil
        import pynvml
        pynvml.nvmlInit()
        self.nvml, self.psutil = pynvml, psutil
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval, self.rows, self.stop_flag = interval, [], threading.Event()

    def run(self):
        disk0, t_prev = self.psutil.disk_io_counters(), time.time()
        self.psutil.cpu_percent(None)
        while not self.stop_flag.wait(self.interval):
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            disk, t = self.psutil.disk_io_counters(), time.time()
            self.rows.append(dict(gpu_util=util.gpu, gpu_mem_util=util.memory, gpu_used_bytes=int(mem.used),
                                  cpu_percent=self.psutil.cpu_percent(None),
                                  ram_used_bytes=int(self.psutil.virtual_memory().used),
                                  disk_read_bps=(disk.read_bytes-disk0.read_bytes)/(t-t_prev),
                                  disk_write_bps=(disk.write_bytes-disk0.write_bytes)/(t-t_prev)))
            disk0, t_prev = disk, t

    def summary(self):
        import numpy as np
        if not self.rows:
            return {}
        keys = self.rows[0].keys()
        return {f"mean_{k}": float(np.mean([r[k] for r in self.rows])) for k in keys} | {
            f"max_{k}": float(np.max([r[k] for r in self.rows])) for k in keys} | {"samples": len(self.rows)}


def run_config(name, args, work):
    tag = f"{name}_{int(time.time())}"
    d = work/tag
    d.mkdir(parents=True)
    go = d/"go"
    procs = []
    for i, c in enumerate(name):
        cmd = [sys.executable, __file__, "worker", "--arch", SHORT[c], "--csv", args.csv,
               "--seconds", str(args.seconds), "--warmup-steps", str(args.warmup_steps),
               "--ready", str(d/f"ready{i}"), "--go", str(go), "--out", str(d/f"result{i}.json")]
        log = open(d/f"worker{i}.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env=os.environ | {"PYTHONUNBUFFERED": "1", "TORCHINDUCTOR_COMPILE_THREADS": "4",
                                                        "OMP_NUM_THREADS": "4"}), log))
    deadline = time.time()+args.ready_timeout
    while not all((d/f"ready{i}").exists() for i in range(len(name))):
        if any(p.poll() not in (None, 0) for p, _ in procs) or time.time() > deadline:
            for p, _ in procs:
                p.kill()
            return dict(config=name, error="worker failed before the barrier",
                        logs=[(d/f"worker{i}.log").read_text()[-2000:] for i in range(len(name))])
        time.sleep(0.2)
    sampler = Sampler()
    sampler.start()
    go.touch()
    codes = [p.wait() for p, _ in procs]
    sampler.stop_flag.set()
    sampler.join()
    for _, log in procs:
        log.close()
    if any(codes):
        return dict(config=name, error=f"exit codes {codes}",
                    logs=[(d/f"worker{i}.log").read_text()[-2000:] for i in range(len(name))])
    jobs = [json.loads((d/f"result{i}.json").read_text()) for i in range(len(name))]
    return dict(config=name, archs=[SHORT[c] for c in name], jobs=jobs, system=sampler.summary())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode")
    w = sub.add_parser("worker")
    w.add_argument("--arch", required=True)
    w.add_argument("--csv", required=True)
    w.add_argument("--seconds", type=float, default=45)
    w.add_argument("--warmup-steps", type=int, default=60)
    w.add_argument("--ready", required=True)
    w.add_argument("--go", required=True)
    w.add_argument("--out", required=True)
    c = sub.add_parser("controller")
    c.add_argument("--csv", required=True)
    c.add_argument("--out", default=str(ROOT/"benchmarks"))
    c.add_argument("--configs", default=",".join(CONFIGS))
    c.add_argument("--seconds", type=float, default=45)
    c.add_argument("--warmup-steps", type=int, default=60)
    c.add_argument("--ready-timeout", type=float, default=900)
    args = p.parse_args(argv)
    if args.mode == "worker":
        return worker(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT/"runs/cls/benchmark_work"
    work.mkdir(parents=True, exist_ok=True)
    results_path = out/"concurrency_benchmark_raw.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {r["config"] for r in results if "error" not in r}
    for name in args.configs.split(","):
        if name in done:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] config {name}", flush=True)
        r = run_config(name, args, work)
        results = [x for x in results if x["config"] != name]+[r]
        results_path.write_text(json.dumps(results, indent=2)+"\n")
        if "error" in r:
            print(f"   ERROR {r['error']}", flush=True)
        else:
            print("   " + "  ".join(f"{j['arch']}={j['samples_per_second']:.0f}/s" for j in r["jobs"])
                  + f"  gpu_util={r['system'].get('mean_gpu_util', 0):.0f}%", flush=True)


if __name__ == "__main__":
    main()
