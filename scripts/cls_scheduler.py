"""Registry-driven job scheduler for one GPU (run it inside tmux).

  python scripts/cls_scheduler.py --stages multi --csv ~/data/BTC_L10_gate_1y.csv

* The registry (runs/cls/registry.json) holds every job and its status:
  planned -> running -> completed | failed. Each launch appends an attempt record
  (start, end, pid, exit code, log). A failed job is re-queued up to --max-attempts
  times; the job runner resumes from its last epoch-boundary checkpoint, and the job is
  flagged `retried`.
* Admission follows benchmarks/training_schedule.json (derived from the measured
  concurrency benchmark): a job is admitted while the summed load of the running jobs
  stays within `capacity`, the number of jobs within `max_concurrent`, and NVML free
  memory above the job's measured peak reserved VRAM plus a safety margin.
* Within the admitted stages, jobs start longest-first (estimated from the measured
  single-job throughput), so the makespan is not decided by a long job started last.
* Restart-safe: jobs marked running whose process is gone are resolved from their
  run directory's status.json (completed) or failed and re-queued.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cls.jobs import Registry, now, run_config

SAMPLES = {1: dict(train=333177, validation=332849), 2: dict(train=666059, validation=332433),
           3: dict(train=998525, validation=334350)}
EVAL_SPEEDUP = 3.0
STOP = False


def handle_signal(signum, frame):
    global STOP
    STOP = True


def estimate_seconds(job, policy):
    sps = policy["single_job_samples_per_second"][job["arch"]]
    counts = SAMPLES[job["fold"]]
    return job["training"]["epochs"]*(counts["train"]+counts["validation"]/EVAL_SPEEDUP)/sps


def gpu_free_bytes():
    try:
        import pynvml
        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free)
    except Exception:
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except OSError:
        return False


def run_state(job):
    path = ROOT/job["run_dir"]/"status.json"
    return json.loads(path.read_text()).get("state") if path.exists() else None


def launch(job, args, log_dir):
    run_dir = ROOT/job["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir/"run_config.json"
    cfg = run_config(job, args.csv)
    if cfg_path.exists():
        previous = json.loads(cfg_path.read_text())
        if {k: v for k, v in previous.items() if k != "csv_path"} != {k: v for k, v in cfg.items() if k != "csv_path"}:
            raise RuntimeError(f"{job['id']}: existing run_config.json differs from the registry job")
    cfg_path.write_text(json.dumps(cfg, indent=2)+"\n")
    log = log_dir/f"{job['id']}.log"
    handle = log.open("a")
    handle.write(f"\n===== {now()} attempt {len(job['attempts'])+1} =====\n")
    handle.flush()
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1",
                        "TORCHINDUCTOR_COMPILE_THREADS": "4", "OMP_NUM_THREADS": "4"}
    process = subprocess.Popen([sys.executable, str(ROOT/"scripts/cls_run.py"), "--config", str(cfg_path)],
                               stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                               start_new_session=True)
    return process, handle, str(log)


def prune(job):
    """After a completed job: the resume-only folder (weights + optimizer) is not needed."""
    last = ROOT/job["run_dir"]/"last"
    if last.exists():
        shutil.rmtree(last)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", default=str(ROOT/"runs/cls/registry.json"))
    p.add_argument("--csv", required=True)
    p.add_argument("--stages", required=True, help="Comma list; only these stages are admitted")
    p.add_argument("--policy", default=str(ROOT/"benchmarks/training_schedule.json"))
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--stagger", type=float, default=30.0, help="Seconds between launches")
    p.add_argument("--safety-bytes", type=int, default=4*1024**3)
    args = p.parse_args(argv)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    policy = json.loads(Path(args.policy).read_text())["policy"]
    registry = Registry(args.registry)
    log_dir = ROOT/"runs/cls/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    children = {}   # id -> (process, handle)
    last_launch = 0.0
    print(f"[{now()}] scheduler start stages={stages} policy={policy}", flush=True)

    # Resolve jobs left running by a previous scheduler.
    with registry.locked() as data:
        for job in data["jobs"]:
            if job["status"] == "running":
                attempt = job["attempts"][-1]
                if alive(attempt["pid"]):
                    print(f"  adopting live {job['id']} pid={attempt['pid']}", flush=True)
                    continue
                state = run_state(job)
                done = state == "completed" and (ROOT/job["run_dir"]/"run_summary.json").exists()
                attempt.update(end=now(), outcome="completed" if done else "lost")
                job["status"] = "completed" if done else "failed"
        for job in data["jobs"]:
            if job["status"] == "completed":
                prune(job)   # idempotent: clears last/ left by a run that ended while we were down

    while not STOP:
        with registry.locked() as data:
            jobs = [j for j in data["jobs"] if j["stage"] in stages]
            # 1. reap
            for job in jobs:
                if job["status"] != "running":
                    continue
                attempt = job["attempts"][-1]
                child = children.get(job["id"])
                if child is not None:
                    code = child[0].poll()
                    if code is None:
                        continue
                    child[1].close()
                    children.pop(job["id"])
                elif alive(attempt["pid"]):
                    continue
                else:
                    code = None
                state = run_state(job)
                ok = state == "completed" and (ROOT/job["run_dir"]/"run_summary.json").exists() and code in (0, None)
                attempt.update(end=now(), returncode=code, outcome="completed" if ok else "failed")
                if ok:
                    job["status"] = "completed"
                    prune(job)
                    print(f"[{now()}] completed {job['id']}", flush=True)
                else:
                    job["status"] = "failed"
                    print(f"[{now()}] FAILED {job['id']} code={code} state={state}", flush=True)
            # 2. retry
            for job in jobs:
                if job["status"] == "failed" and len(job["attempts"]) < args.max_attempts:
                    job["status"], job["retried"] = "planned", True
                    print(f"[{now()}] re-queued {job['id']} (attempt {len(job['attempts'])+1})", flush=True)
            running = [j for j in jobs if j["status"] == "running"]
            planned = [j for j in jobs if j["status"] == "planned"]
            if not running and not planned:
                break
            # 3. admit (longest first)
            # A job in its CPU-only writing phase no longer occupies the GPU.
            gpu_jobs = [j for j in running if run_state(j) != "writing"]
            load = sum(policy["load"][j["arch"]] for j in gpu_jobs)
            if planned and time.time()-last_launch >= args.stagger:
                planned.sort(key=lambda j: -estimate_seconds(j, policy))
                free = gpu_free_bytes()
                for job in planned:
                    need = policy["load"][job["arch"]]
                    if len(gpu_jobs) >= policy["max_concurrent"] or load+need > policy["capacity"]+1e-9:
                        continue
                    vram = policy["peak_reserved_bytes"][job["arch"]]
                    if free is not None and free < vram+args.safety_bytes:
                        continue
                    try:
                        process, handle, log = launch(job, args, log_dir)
                    except Exception as error:  # a bad job must not take the sweep down
                        job["attempts"].append(dict(start=now(), end=now(), outcome="launch-error", error=repr(error)))
                        job["status"] = "failed"
                        print(f"[{now()}] LAUNCH ERROR {job['id']}: {error!r}", flush=True)
                        continue
                    children[job["id"]] = (process, handle)
                    job["attempts"].append(dict(start=now(), pid=process.pid, log=log))
                    job["status"] = "running"
                    last_launch = time.time()
                    print(f"[{now()}] start {job['id']} pid={process.pid} load={load+need:.2f}/"
                          f"{policy['capacity']} gpu_jobs={len(gpu_jobs)+1} est={estimate_seconds(job, policy)/3600:.2f}h",
                          flush=True)
                    break
        time.sleep(args.poll)

    if STOP:
        print(f"[{now()}] stop requested; running jobs keep going and will be adopted on restart", flush=True)
        return 1
    data = registry.read()
    failed = [j["id"] for j in data["jobs"] if j["stage"] in stages and j["status"] == "failed"]
    print(f"[{now()}] stages {stages} finished; failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
