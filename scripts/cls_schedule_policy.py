"""Turn the raw concurrency benchmark into the sweep's scheduling policy.

  python scripts/cls_schedule_policy.py

Reads benchmarks/concurrency_benchmark_raw.json and writes
  benchmarks/concurrency_benchmark.json  single-job and per-configuration summary
  benchmarks/training_schedule.json      the admission policy scripts/cls_scheduler.py uses

Accounting. Progress of a job is measured in isolated-job units: its samples/s divided
by the samples/s of the same architecture running alone, so serial execution scores
1.0 per second. For each architecture a, k_a is the number of concurrent same-arch jobs
that reaches its best throughput R_a (smallest k within 3% of the best). A mixed
configuration that delivers p_a units/s of each architecture a is only worth running if
it beats time-slicing the exclusive groups, i.e. if

    efficiency = sum_a (p_a / R_a)   >   1

(the seconds the exclusive groups would need to deliver one second of the mix).

The policy is then a load model: load_a = 1/k_a, capacity 1.0 (exclusive groups), unless
some measured mixed configuration has efficiency > 1.02, in which case the capacity is
raised to admit the best such configuration.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SHORT = dict(T="moderntcn", F="transformer", L="ofi_lstm")


def main():
    raw = json.loads((ROOT/"benchmarks/concurrency_benchmark_raw.json").read_text())
    ok = {r["config"]: r for r in raw if "error" not in r}
    failed = [dict(config=r["config"], error=r["error"]) for r in raw if "error" in r]
    single = {}
    for c in "TFL":
        job = ok[c]["jobs"][0]
        single[SHORT[c]] = job | dict(gpu_util=ok[c]["system"]["mean_gpu_util"])
    configs = []
    for name, r in ok.items():
        sps = [j["samples_per_second"] for j in r["jobs"]]
        units = {}
        for s, j in zip(sps, r["jobs"]):
            units[j["arch"]] = units.get(j["arch"], 0.0)+s/single[j["arch"]]["samples_per_second"]
        configs.append(dict(config=name, archs=r["archs"], samples_per_second=sps, units_per_second=units,
                            normalized_throughput=sum(units.values()), total_samples_per_second=sum(sps),
                            mean_gpu_util=r["system"].get("mean_gpu_util", 0),
                            max_gpu_used_bytes=r["system"].get("max_gpu_used_bytes", 0),
                            mean_cpu_percent=r["system"].get("mean_cpu_percent", 0),
                            max_ram_used_bytes=r["system"].get("max_ram_used_bytes", 0),
                            mean_disk_read_bps=r["system"].get("mean_disk_read_bps", 0),
                            mean_disk_write_bps=r["system"].get("mean_disk_write_bps", 0),
                            peak_reserved_bytes=[j["peak_reserved_bytes"] for j in r["jobs"]],
                            process_cpu_percent=[j["process_cpu_percent"] for j in r["jobs"]]))
    configs.sort(key=lambda c: (len(c["config"]), c["config"]))
    best_k, rate, same = {}, {}, {}
    for c in "TFL":
        runs = [x for x in configs if set(x["config"]) == {c}]
        same[SHORT[c]] = {len(x["config"]): round(x["normalized_throughput"], 3) for x in runs}
        top = max(x["normalized_throughput"] for x in runs)
        best_k[SHORT[c]] = min(len(x["config"]) for x in runs if x["normalized_throughput"] >= 0.97*top)
        rate[SHORT[c]] = max(x["normalized_throughput"] for x in runs if len(x["config"]) == best_k[SHORT[c]])
    for x in configs:
        # Seconds the exclusive groups would need to deliver one second of this configuration.
        x["efficiency_vs_exclusive_groups"] = sum(u/rate[a] for a, u in x["units_per_second"].items())
    mixed = [x for x in configs if len(set(x["config"])) > 1]
    load = {a: 1.0/k for a, k in best_k.items()}
    winners = [x for x in mixed if x["efficiency_vs_exclusive_groups"] > 1.02]
    if winners:
        best = max(winners, key=lambda x: x["efficiency_vs_exclusive_groups"])
        capacity = sum(load[SHORT[ch]] for ch in best["config"])
        max_concurrent = max(len(best["config"]), max(best_k.values()))
    else:
        best, capacity, max_concurrent = None, 1.0, max(best_k.values())
    policy = dict(max_concurrent=int(max_concurrent), capacity=float(capacity), load=load,
                  peak_reserved_bytes={a: int(v["peak_reserved_bytes"]*1.25)+int(v["resident_feature_bytes"])
                                       for a, v in single.items()},
                  single_job_samples_per_second={a: v["samples_per_second"] for a, v in single.items()})
    eff = {x["config"]: round(x["efficiency_vs_exclusive_groups"], 3) for x in mixed}
    rationale = (
        f"Same-architecture scaling, normalized throughput by number of jobs: {json.dumps(same)}. "
        f"ModernTCN and the Transformer saturate the GPU alone (mean util {single['moderntcn']['gpu_util']:.0f}% / "
        f"{single['transformer']['gpu_util']:.0f}%) and a second heavy job only loses throughput; OFI-LSTM is "
        f"latency-bound alone ({single['ofi_lstm']['gpu_util']:.0f}% util) and peaks at k={best_k['ofi_lstm']} "
        f"({rate['ofi_lstm']:.2f}x). Mixed configurations reach normalized throughput up to "
        f"{max(x['normalized_throughput'] for x in mixed):.2f}x, but they buy it by slowing the heavy job; against "
        f"time-slicing the exclusive groups their efficiency is {eff} (1.00 = equal). "
        + (f"The best mixed configuration {best['config']} beats exclusive groups, so capacity {capacity:.2f} admits it."
           if best else "No mixed configuration beats exclusive groups, so the policy runs one ModernTCN or one "
           f"Transformer alone, or up to {best_k['ofi_lstm']} OFI-LSTM jobs together (capacity 1.0).")
        + " Batch size and every hyperparameter stay fixed; only the number of concurrent jobs is chosen.")
    out = dict(single_job=single, configurations=configs, failed_configurations=failed,
               same_architecture_scaling=same, best_k=best_k, exclusive_group_rate=rate)
    (ROOT/"benchmarks/concurrency_benchmark.json").write_text(json.dumps(out, indent=2)+"\n")
    admissible = [dict(config=x["config"], load=round(sum(load[SHORT[ch]] for ch in x["config"]), 3),
                       normalized=round(x["normalized_throughput"], 3),
                       efficiency_vs_exclusive_groups=round(x["efficiency_vs_exclusive_groups"], 3),
                       admitted=sum(load[SHORT[ch]] for ch in x["config"]) <= capacity+1e-9
                       and len(x["config"]) <= max_concurrent) for x in configs]
    schedule = dict(policy=policy, rationale=rationale, admissibility_of_measured_configurations=admissible,
                    ordering="longest estimated job first within the admitted stage",
                    oom_policy="1) fewer concurrent jobs 2) batch size only if unavoidable 3) implementation-level "
                               "memory work; never shrink the architecture",
                    gpu=dict(name="NVIDIA GeForce RTX 4090", reported_total_bytes=50770362368,
                             note="the card reports 47.4 GiB, not the 24 GB the prompt assumed; memory never bound"))
    (ROOT/"benchmarks/training_schedule.json").write_text(json.dumps(schedule, indent=2)+"\n")
    print(rationale)
    for a in admissible:
        print(a)


if __name__ == "__main__":
    sys.exit(main())
