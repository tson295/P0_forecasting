"""Background GPU sampler for reports/vast/gpu_usage.csv (prompt.md section Z)."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import sys
import time

USAGE_COLUMNS = ["timestamp", "label", "gpu_util_percent", "memory_used_mb", "memory_free_mb",
                 "memory_total_mb", "temperature_c", "power_draw_w", "sm_clock_mhz", "process_count"]
PROCESS_COLUMNS = ["timestamp", "pid", "process_name", "used_memory_mb", "label"]
GPU_FIELDS = ("utilization.gpu,memory.used,memory.free,memory.total,"
              "temperature.gpu,power.draw,clocks.sm")
GPU_KEYS = ["gpu_util_percent", "memory_used_mb", "memory_free_mb", "memory_total_mb",
            "temperature_c", "power_draw_w", "sm_clock_mhz"]
MIB = 2**20
FALLBACK_AFTER = 3  # consecutive NVML failures before switching backends


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def proc_comm(pid):
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:  # driver pids come from the host namespace inside a container
        return "unknown"


def number(text):
    # nvidia-smi prints [N/A] / [Not Supported] for unsupported sensors; keep the cell empty.
    try:
        value = float(text)
    except ValueError:
        return ""
    return int(value) if value.is_integer() else round(value, 1)


def call(fn, *args):
    # One unsupported sensor must cost a single cell, not the whole row for the rest of the run.
    try:
        return fn(*args)
    except Exception:
        return None


def blank(value):
    return "" if value is None else value


def mib(value):
    return "" if value is None else round(value/MIB, 1)


class NvmlSampler:
    backend = "pynvml"

    def __init__(self, index):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.nvml.nvmlDeviceGetUtilizationRates(self.handle)  # fail fast on a dead/denied device

    def process_name(self, pid):
        try:
            value = self.nvml.nvmlSystemGetProcessName(pid)
        except Exception:  # NVML hides names of processes it cannot resolve for us
            return proc_comm(pid)
        value = value.decode() if isinstance(value, bytes) else value
        return proc_comm(pid) if value.startswith("[") else value  # "[Not Found]" style placeholder

    def metrics(self):
        nvml, handle = self.nvml, self.handle
        memory = call(nvml.nvmlDeviceGetMemoryInfo, handle)
        util = call(nvml.nvmlDeviceGetUtilizationRates, handle)
        power = call(nvml.nvmlDeviceGetPowerUsage, handle)
        row = dict(gpu_util_percent=blank(util and util.gpu),
                   memory_used_mb=mib(memory and memory.used),
                   memory_free_mb=mib(memory and memory.free),
                   memory_total_mb=mib(memory and memory.total),
                   temperature_c=blank(call(nvml.nvmlDeviceGetTemperature, handle,
                                            nvml.NVML_TEMPERATURE_GPU)),
                   power_draw_w="" if power is None else round(power/1000, 1),
                   sm_clock_mhz=blank(call(nvml.nvmlDeviceGetClockInfo, handle, nvml.NVML_CLOCK_SM)))
        if all(cell == "" for cell in row.values()):
            raise RuntimeError("NVML returned no usable metrics")  # let the caller switch backends
        return row

    def processes(self):
        nvml, handle = self.nvml, self.handle
        running = {}  # keyed by pid: a process can appear in both lists
        for info in (nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                     + nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)):
            running[info.pid] = dict(pid=info.pid, process_name=self.process_name(info.pid),
                                     used_memory_mb=mib(getattr(info, "usedGpuMemory", None)))
        return list(running.values())

    def close(self):
        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass


class SmiSampler:
    backend = "nvidia-smi"

    def __init__(self, index):
        self.index = str(index)
        self.metrics()  # fail fast if nvidia-smi is missing or the index is wrong

    def query(self, what, fields):
        out = subprocess.run(["nvidia-smi", f"--query-{what}={fields}", "-i", self.index,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return [[cell.strip() for cell in row] for row in csv.reader(out.splitlines()) if row]

    def metrics(self):
        rows = self.query("gpu", GPU_FIELDS)
        if not rows:
            raise RuntimeError(f"nvidia-smi returned no row for device {self.index}")
        return dict(zip(GPU_KEYS, (number(cell) for cell in rows[0])))

    def processes(self):
        running = []
        for row in self.query("compute-apps", "pid,process_name,used_memory"):
            if len(row) < 3:
                continue
            pid, name, used = row[0], ",".join(row[1:-1]), row[-1]  # names may contain commas
            # nvidia-smi prints [Not Found] when the pid lives in another namespace; match NVML.
            running.append(dict(pid=pid, process_name=proc_comm(pid) if name.startswith("[") else name,
                                used_memory_mb=number(used)))
        return running

    def close(self):
        pass


def open_sampler(index):
    try:
        return NvmlSampler(index)
    except Exception as nvml_error:
        print(f"pynvml unavailable ({nvml_error!r}); falling back to nvidia-smi",
              file=sys.stderr, flush=True)
    try:
        return SmiSampler(index)
    except Exception as smi_error:  # a bad --device deserves one line, not two tracebacks
        raise SystemExit(f"no GPU backend for device {index}: {smi_error!r}")


def swap_backend(sampler, index):
    # run_training.py discards our stderr, so a wedged NVML has to repair itself, not just complain.
    try:
        replacement = SmiSampler(index)
    except Exception as error:
        print(f"nvidia-smi fallback unavailable: {error!r}", file=sys.stderr, flush=True)
        return sampler
    sampler.close()
    print(f"gpu_monitor backend -> {replacement.backend} after {FALLBACK_AFTER} failed samples",
          file=sys.stderr, flush=True)
    return replacement


def open_csv(path, columns, append):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists() and path.stat().st_size
    if append and existing:
        with open(path) as handle:
            header = handle.readline().strip()
        if header != ",".join(columns):  # otherwise the appended rows silently misalign
            raise SystemExit(f"{path}: header {header!r} does not match this schema; drop --append")
    handle = open(path, "a" if append else "w", newline="")
    writer = csv.DictWriter(handle, columns, extrasaction="ignore")
    if not (append and existing):
        writer.writeheader()
        handle.flush()
    return handle, writer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("reports/vast/gpu_usage.csv"))
    p.add_argument("--process-output", type=Path, help="default: gpu_processes.csv beside --output")
    p.add_argument("--interval", type=float, default=3.0, help="sampling period, 2-5 seconds")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--label", default="", help="free-text tag for the current phase")
    p.add_argument("--append", action="store_true")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()
    if not 2.0 <= args.interval <= 5.0:
        p.error("--interval must be within 2-5 seconds (prompt.md section Z)")
    process_output = args.process_output or args.output.with_name("gpu_processes.csv")

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))  # training teardown kills us; exit clean
    sampler = open_sampler(args.device)
    usage_file, usage = open_csv(args.output, USAGE_COLUMNS, args.append)
    process_file, processes = open_csv(process_output, PROCESS_COLUMNS, args.append)
    print(f"gpu_monitor backend={sampler.backend} device={args.device} interval={args.interval}s "
          f"-> {args.output} + {process_output}", flush=True)
    deadline = time.monotonic()+args.duration if args.duration is not None else None
    samples = errors = streak = 0
    try:
        while not stop and (deadline is None or time.monotonic() < deadline):
            stamp = now_iso()
            try:
                row = sampler.metrics()
            except Exception as error:  # a transient NVML/smi hiccup must never end the monitor
                errors += 1
                streak += 1
                print(f"{stamp} sample failed: {error!r}", file=sys.stderr, flush=True)
                if streak == FALLBACK_AFTER and sampler.backend == NvmlSampler.backend:
                    sampler = swap_backend(sampler, args.device)
            else:
                streak = 0
                try:
                    running = sampler.processes()
                except Exception as error:  # the PID mapping is optional, the metrics row is not
                    running = None
                    print(f"{stamp} process query failed: {error!r}", file=sys.stderr, flush=True)
                usage.writerow(row | dict(timestamp=stamp, label=args.label,
                                          process_count="" if running is None else len(running)))
                for entry in running or ():
                    processes.writerow(entry | dict(timestamp=stamp, label=args.label))
                usage_file.flush()
                process_file.flush()  # flush per sample: a killed monitor still leaves a full CSV
                samples += 1
            target = time.monotonic()+args.interval
            while not stop and (remaining := target-time.monotonic()) > 0:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(min(0.2, remaining))  # short naps keep SIGTERM latency under a second
    finally:
        usage_file.close()
        process_file.close()
        sampler.close()
    print(f"gpu_monitor stopped: {samples} samples, {errors} errors", flush=True)
    return 1 if errors and not samples else 0  # an all-failed run must not look successful


if __name__ == "__main__":
    sys.exit(main())
