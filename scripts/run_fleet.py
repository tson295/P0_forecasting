"""Saturate one GPU with a fleet of frozen training runs.

Longest-processing-time-first over a dynamic work queue: the makespan of N independent
jobs on one device is bounded below by the longest single job, so the long ones must
start first and the short ones fill the gaps behind them. Admission is gated on measured
VRAM, never on a guess, and each job keeps its own frozen batch size and hyperparameters.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.training.checkpoint import write_json

LEARNED = ("ofi_lstm", "hfformer", "patchtst", "moderntcn", "lit")
EVAL_SPEEDUP = 2.5          # validation is forward-only; measured ratio on this box
SAFETY_BYTES = 3*1024**3    # keep at least this much VRAM free, as the contract requires


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def gpu_total_bytes():
    import torch
    return int(torch.cuda.get_device_properties(0).total_memory)


def build_jobs(args, benchmark):
    """One job per (model, fold), costed from the measured single-job throughput."""
    jobs = []
    for model in args.models:
        for fold in args.folds:
            run = args.run_template.format(model=model, fold=fold)
            path = Path("configs")/f"{run}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            config = Config.load(path)
            measured = benchmark["models"][model]
            counts = args.sample_counts[str(fold)]
            work = counts["train"]+counts["validation"]/EVAL_SPEEDUP
            jobs.append(dict(
                model=model, fold=fold, run_name=run, config=str(path),
                estimated_seconds=config.training.epochs*work/measured["samples_per_second"],
                vram_bytes=int(measured["peak_reserved_bytes"]*args.vram_headroom),
                train_samples=counts["train"], validation_samples=counts["validation"],
                epochs=config.training.epochs, batch_size=config.training.batch_size,
                compile=bool(config.training.compile)))
    # Longest first: the last job to start decides the makespan, so it must be a short one.
    jobs.sort(key=lambda j: -j["estimated_seconds"])
    return jobs


def launch(job, args, log_directory):
    command = [sys.executable, "train.py", "--config", job["config"], "--csv", args.csv,
               "--num-workers", str(args.num_workers), *args.extra]
    last = Path("checkpoints")/job["model"]/job["run_name"]/"last"
    if job.get("resume"):
        command += ["--resume", str(last)]
    log = log_directory/f"{job['run_name']}.log"
    handle = log.open("a", encoding="utf-8")
    handle.write(f"\n===== {now()} launch {' '.join(command)} =====\n")
    handle.flush()
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               env=os.environ | {"CUDA_VISIBLE_DEVICES": "0",
                                                 "PYTHONUNBUFFERED": "1"})
    return dict(job, pid=process.pid, started=now(), started_monotonic=time.monotonic(),
                log=str(log), _process=process, _handle=handle)


def reap(running, records, args, log_directory):
    """Collect finished jobs; resume one that crashed with a usable `last` checkpoint."""
    still, requeue = [], []
    for entry in running:
        code = entry["_process"].poll()
        if code is None:
            still.append(entry)
            continue
        entry["_handle"].close()
        entry["returncode"] = code
        entry["finished"] = now()
        entry["seconds"] = time.monotonic()-entry.pop("started_monotonic")
        process, _ = entry.pop("_process"), entry.pop("_handle")
        del process
        if code != 0:
            last = Path("checkpoints")/entry["model"]/entry["run_name"]/"last"
            resumable = (last/"trainer_state.pt").exists()
            print(f"[{now()}] {entry['run_name']} exited {code}; resumable={resumable}", flush=True)
            if resumable and not entry.get("resume") and not args.no_resume:
                requeue.append(dict({k: v for k, v in entry.items()
                                     if not k.startswith(("pid", "started", "finished",
                                                          "seconds", "returncode", "log"))},
                                    resume=True))
            else:
                raise RuntimeError(f"{entry['run_name']} failed (exit {code}); see {entry['log']}")
        records.append(entry)
    return still, requeue


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=os.environ.get("LOB_CSV", "BTC_L10_gate_1y.csv"))
    p.add_argument("--run-template", default="{model}_wf3_f{fold}")
    p.add_argument("--models", default=",".join(LEARNED))
    p.add_argument("--folds", default="1,2,3")
    p.add_argument("--benchmark", default="reports/vast_wf3/single_job_benchmark.json")
    p.add_argument("--reports", default="reports/vast_wf3")
    p.add_argument("--log-directory", default="runs/logs")
    p.add_argument("--max-concurrent", type=int, default=0, help="0 = decide from VRAM alone")
    p.add_argument("--vram-headroom", type=float, default=1.25,
                   help="Multiplier on measured peak reserved VRAM when admitting a job")
    p.add_argument("--safety-bytes", type=int, default=SAFETY_BYTES)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--stagger-seconds", type=float, default=20.0,
                   help="Spacing between launches so CSV loads do not peak together")
    p.add_argument("--monitor-interval", type=float, default=5.0)
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("extra", nargs="*")
    args = p.parse_args(argv)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.folds = [int(f) for f in args.folds.split(",")]

    benchmark = json.loads(Path(args.benchmark).read_text())
    # Sample counts come from the configs themselves so the estimate cannot drift from reality.
    args.sample_counts = {}
    for fold in args.folds:
        manifest = Path(args.reports)/f"fold_{fold}_samples.json"
        args.sample_counts[str(fold)] = json.loads(manifest.read_text())

    jobs = build_jobs(args, benchmark)
    total = gpu_total_bytes()
    budget = total-args.safety_bytes
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    log_directory = Path(args.log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    plan = dict(created=now(), csv=args.csv, gpu_total_bytes=total, vram_budget_bytes=budget,
                safety_bytes=args.safety_bytes, vram_headroom=args.vram_headroom,
                max_concurrent=args.max_concurrent or None, num_workers=args.num_workers,
                policy="longest-processing-time-first over a dynamic queue; admission gated on "
                       "measured peak reserved VRAM; batch size and hyperparameters stay frozen",
                estimated_serial_seconds=sum(j["estimated_seconds"] for j in jobs),
                estimated_makespan_lower_bound_seconds=max(j["estimated_seconds"] for j in jobs),
                jobs=[{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs])
    write_json(reports/"fleet_plan.json", plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}, indent=2), flush=True)
    for j in jobs:
        print(f"  {j['run_name']:22s} est {j['estimated_seconds']/3600:6.2f} h  "
              f"vram {j['vram_bytes']/2**30:5.2f} GiB  compile={j['compile']}", flush=True)
    if args.dry_run:
        return plan

    monitor = None
    if not args.no_monitor:
        monitor = subprocess.Popen(
            [sys.executable, "scripts/gpu_monitor.py", "--append", "--interval",
             str(args.monitor_interval), "--label", "fleet_wf3",
             "--output", str(reports/"gpu_usage.csv"),
             "--process-output", str(reports/"gpu_processes.csv")],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    queue, running, records = list(jobs), [], []
    started_at, last_launch = time.monotonic(), 0.0
    try:
        while queue or running:
            used = sum(e["vram_bytes"] for e in running)
            launched = False
            for index, job in enumerate(queue):
                at_cap = args.max_concurrent and len(running) >= args.max_concurrent
                if at_cap or used+job["vram_bytes"] > budget:
                    continue
                if running and time.monotonic()-last_launch < args.stagger_seconds:
                    break
                entry = launch(queue.pop(index), args, log_directory)
                running.append(entry)
                last_launch = time.monotonic()
                print(f"[{now()}] start {entry['run_name']} "
                      f"({len(running)} running, {len(queue)} queued, "
                      f"{(used+job['vram_bytes'])/2**30:.1f}/{budget/2**30:.1f} GiB)", flush=True)
                launched = True
                break
            if not launched:
                time.sleep(5)
            running, requeue = reap(running, records, args, log_directory)
            for job in requeue:
                queue.insert(0, job)
    finally:
        for entry in running:
            entry["_process"].kill()
            entry["_handle"].close()
        if monitor and monitor.poll() is None:
            monitor.send_signal(signal.SIGTERM)
            try:
                monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:
                monitor.kill()
        summary = dict(finished=now(), total_seconds=time.monotonic()-started_at, plan=plan,
                       jobs=records,
                       crashed=[r["run_name"] for r in records if r.get("returncode")],
                       resumed=[r["run_name"] for r in records if r.get("resume")],
                       gpu_usage_csv=str(reports/"gpu_usage.csv"))
        write_json(reports/"fleet_summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("finished", "total_seconds", "crashed", "resumed")},
                         indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
