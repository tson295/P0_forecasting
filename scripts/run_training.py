"""Deterministic schedule runner: frozen runs, GPU monitoring, crash-resume policy.

Concurrency only decides WHICH frozen run executes at the same time. It never changes
batch size, architecture, learning rate, epochs, stride or split.
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
from src.training.checkpoint import write_json

# --suite selects one frozen experiment: its run names, its reports directory (so two suites never
# share a schedule, a summary or a gpu_usage.csv) and its book CSV. Nothing else differs.
SUITES = {"oct2023": dict(runs={"ofi_lstm": "ofi_lstm_60s_base", "hfformer": "hfformer_60s_base",
                                "patchtst": "patchtst_60s_base", "moderntcn": "moderntcn_60s_base",
                                "lit": "lit_60s_base"},
                          reports=Path("reports/vast"), csv="BTCUSDT_L10_oct2023.csv",
                          history_seconds=60),
          "gate1y": dict(runs={"ofi_lstm": "ofi_lstm_490s_gate1y", "hfformer": "hfformer_490s_gate1y",
                               "patchtst": "patchtst_490s_gate1y", "moderntcn": "moderntcn_490s_gate1y",
                               "lit": "lit_490s_gate1y"},
                         reports=Path("reports/vast_gate1y"), csv="BTC_L10_gate_1y.csv",
                         history_seconds=490)}
RUNS, REPORTS = SUITES["oct2023"]["runs"], SUITES["oct2023"]["reports"]


def select_suite(name):
    """Rebind the frozen run names and the reports directory to one experiment."""
    suite = SUITES[name]
    globals()["RUNS"], globals()["REPORTS"] = suite["runs"], suite["reports"]
    return suite


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_groups(args):
    """Benchmark-driven by default; --groups only for an explicit manual override."""
    if args.groups:
        groups = [[m.strip() for m in group.split(",") if m.strip()] for group in args.groups]
        source = "manual --groups override"
    else:
        decision = json.loads((REPORTS/"concurrency_benchmark.json").read_text())["decision"]
        groups = decision.get("recommended_groups") or decision["groups"]
        source = f"{REPORTS}/concurrency_benchmark.json"
    flat = [model for group in groups for model in group]
    if sorted(flat) != sorted(RUNS) or len(flat) != len(set(flat)):
        raise ValueError(f"Schedule must cover each learned model exactly once, got {flat}")
    return groups, source


def launch(model, csv, extra, log_directory, resume=None):
    run = RUNS[model]
    command = [sys.executable, "train.py", "--config", f"configs/{run}.json", "--csv", csv, *extra]
    if resume:
        command += ["--resume", resume]
    log = log_directory/f"{run}.log"
    handle = log.open("a", encoding="utf-8")
    handle.write(f"\n===== {now()} launch {' '.join(command)} =====\n")
    handle.flush()
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    return dict(model=model, run_name=run, pid=process.pid, command=command, log=str(log),
                started=now(), started_monotonic=time.monotonic(),
                resumed_from=resume, process=process, handle=handle)


def wait_for(jobs):
    for job in jobs:
        job["returncode"] = job["process"].wait()
        job["finished"] = now()
        job["seconds"] = time.monotonic()-job.pop("started_monotonic")
        job["handle"].close()
        del job["process"], job["handle"]
    return jobs


def run_group(index, group, args, log_directory, records):
    print(f"[{now()}] group {index}: {', '.join(group)}", flush=True)
    jobs = wait_for([launch(model, args.csv, args.extra, log_directory) for model in group])
    for job in jobs:
        if job["returncode"] == 0:
            continue
        last = Path("checkpoints")/job["model"]/job["run_name"]/"last"
        print(f"[{now()}] {job['run_name']} exited {job['returncode']}; "
              f"resumable={last.exists()}", flush=True)
        # Contract AG: resume from a valid `last`, never restart from epoch 0, never shrink the model.
        if last.exists() and (last/"trainer_state.pt").exists() and not args.no_resume:
            retry = wait_for([launch(job["model"], args.csv, args.extra, log_directory, str(last))])[0]
            retry["is_resume_of"] = job["run_name"]
            records.append(retry)
            if retry["returncode"] != 0:
                raise RuntimeError(f"{job['run_name']} failed again after resume; see {retry['log']}")
        else:
            raise RuntimeError(f"{job['run_name']} failed with no resumable checkpoint; see {job['log']}")
    records.extend(jobs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=sorted(SUITES), default="oct2023",
                   help="Frozen experiment to schedule: run names, reports directory and default CSV")
    p.add_argument("--csv", help="Default: $LOB_CSV, else the suite's book CSV")
    p.add_argument("--groups", action="append",
                   help="Comma-separated models; repeat per group. Default: benchmark decision.")
    p.add_argument("--log-directory", default="runs/logs")
    p.add_argument("--monitor-interval", type=float, default=3.0)
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("extra", nargs="*", help="Extra train.py flags, e.g. --num-workers 6")
    args = p.parse_args(argv)
    suite = select_suite(args.suite)
    args.csv = args.csv or os.environ.get("LOB_CSV") or suite["csv"]

    groups, source = resolve_groups(args)
    log_directory = Path(args.log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    schedule = dict(created=now(), suite=args.suite, decided_by=source, groups=groups, csv=args.csv,
                    runs={model: RUNS[model] for model in RUNS}, extra_train_flags=args.extra,
                    frozen=dict(batch_size=128, epochs=30, seed=42, learning_rate=1e-4,
                                weight_decay=1e-4, optimizer="AdamW", loss="mse",
                                scheduler="CosineAnnealingLR", gradient_clip=1.0,
                                gradient_accumulation=1,
                                history_seconds=suite["history_seconds"], history_rows=49,
                                stride_rows=8),
                    policy="concurrency selects co-execution only; no run parameter changes")
    write_json(REPORTS/"training_schedule.json", schedule)
    print(json.dumps(schedule, indent=2), flush=True)
    if args.dry_run:
        return schedule

    monitor = None
    if not args.no_monitor:
        monitor = subprocess.Popen(
            [sys.executable, "scripts/gpu_monitor.py", "--append",
             "--interval", str(args.monitor_interval), "--label", f"full_training_{args.suite}",
             "--output", str(REPORTS/"gpu_usage.csv")],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    records, started = [], time.monotonic()
    try:
        for index, group in enumerate(groups, 1):
            run_group(index, group, args, log_directory, records)
    finally:
        if monitor and monitor.poll() is None:
            monitor.send_signal(signal.SIGTERM)
            try:
                monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:
                monitor.kill()
        summary = dict(finished=now(), total_seconds=time.monotonic()-started,
                       schedule=schedule, jobs=records,
                       crashed=[j["run_name"] for j in records if j.get("returncode")],
                       resumed=[j["run_name"] for j in records if j.get("is_resume_of")],
                       gpu_usage_csv=str(REPORTS/"gpu_usage.csv"))
        write_json(REPORTS/"final_training_summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("finished", "total_seconds", "crashed", "resumed")},
                         indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
