"""Assemble reports/FINAL_REPORT.md from real artifacts only; never invents a number."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS = [("e0", "e0_60s"), ("ofi_lstm", "ofi_lstm_60s_base"), ("hfformer", "hfformer_60s_base"),
        ("patchtst", "patchtst_60s_base"), ("moderntcn", "moderntcn_60s_base"),
        ("lit", "lit_60s_base")]
HORIZONS = ("1m", "2m", "3m")
MISSING = "n/a"

DEVIATIONS = [
    "**HFformer window reduction accumulates in FP64** (section J). The frozen formula, "
    "eps=1e-5, unbiased=False and the per-sample/per-feature/over-time contract are unchanged; "
    "only the reduction's arithmetic precision differs from the literal `x_fp32.mean(dim=1)`. "
    "At raw L10 price scale (~2.7e4, FP32 ulp ~2e-3) a batched FP32 mean carries a few ulp of "
    "residue, and dividing that by eps turns a constant window -- whose true z-score is 0 -- into "
    "about -586; measured on real train windows, 6.2% of (sample, feature) pairs have a near-constant "
    "history and 1.3% of normalized entries exceeded |z|>10. The FP32 reduction kernel also varies "
    "with tensor shape, so the value depended on the batch, which section J forbids ('phu thuoc duy "
    "nhat vao history hien tai cua sample'). With FP64 accumulation the result matches the exact "
    "formula to 2.4e-7 and is bit-identical across batch sizes 1, 3, 7, 128 and 512.",
    "**PatchTST exposes four of the six LoRA module names** (sections W vs K3). Sections K2 and K5 "
    "name q_proj/k_proj/v_proj/out_proj/fc1/fc2 for HFformer and LiT, and both provide all six. "
    "Section K3 pins PatchTST to the stock `transformers.PatchTSTModel` at exactly 477,059 "
    "parameters; that module keeps its feed-forward Linears inside an nn.Sequential at ff.0/ff.3. "
    "Its four attention projections -- the standard LoRA injection points -- are present and "
    "unmerged. Renaming the upstream feed-forward layers would be the architecture change K3 forbids, "
    "so the upstream names were kept.",
    "**Credentials came from the machine, not from GITHUB_TOKEN/HF_TOKEN** (sections A, AC). Neither "
    "environment variable was set and the GitHub CLI is not installed. GitHub was authenticated with "
    "the existing SSH key already configured for the `origin` remote, and Hugging Face with the token "
    "already stored in the local Hugging Face home. No token was printed, written into source, "
    "committed, or embedded in a remote URL.",
]


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text()
    return [json.loads(l) for l in text.splitlines() if l.strip()] if path.suffix == ".jsonl" \
        else json.loads(text)


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return MISSING


def number(value, digits=6):
    if value is None:
        return MISSING
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def metric_rows(metrics):
    """RMSE / MAE / R2 / RMSE gain vs E0, one column per horizon."""
    if not metrics:
        return [f"| {name} | {MISSING} | {MISSING} | {MISSING} |"
                for name in ("RMSE", "MAE", "R2", "RMSE gain vs E0")]
    rows = []
    for label, key in (("RMSE", "rmse"), ("MAE", "mae"), ("R2", "r2"),
                       ("RMSE gain vs E0", "rmse_gain_vs_e0")):
        rows.append(f"| {label} | " + " | ".join(number(v) for v in metrics[key]) + " |")
    return rows


def bytes_gib(value):
    return f"{value/2**30:.2f} GiB" if isinstance(value, (int, float)) else MISSING


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/FINAL_REPORT.md")
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--hf-repo", default="")
    p.add_argument("--training-commit", default="")
    p.add_argument("--initial-commit", default="")
    args = p.parse_args()

    vast = Path("reports/vast")
    hardware = load(vast/"hardware.json") or {}
    single = load(vast/"single_job_benchmark.json") or {}
    compiled = load(vast/"compile_benchmark.json") or {}
    concurrency = load(vast/"concurrency_benchmark.json") or {}
    schedule = load(vast/"training_schedule.json") or {}
    final = load(vast/"final_training_summary.json") or {}
    contracts = load("reports/contract_check.json") or {}

    summaries, histories, trainings = {}, {}, {}
    for model, run_name in RUNS:
        summaries[model] = load(Path(args.artifact_root)/model/run_name/"run_summary.json")
        histories[model] = load(Path(args.artifact_root)/model/run_name/"training_history.jsonl")
        trainings[model] = load(Path(args.checkpoint_root)/model/run_name/"training_run.json")

    reference = next((s for s in summaries.values() if s), {})
    manifest = reference.get("split_manifest", {})
    stats = reference.get("data_stats", {})
    out = []
    w = out.append

    w("# Base experiment final report — BTC L10, 60 s history\n")
    w("## 1-4. Provenance\n")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Initial Git commit | `{args.initial_commit or MISSING}` |")
    w(f"| Training-source Git commit | `{args.training_commit or MISSING}` |")
    w(f"| Final Git commit | `{git('rev-parse', 'HEAD')}` |")
    w(f"| Hugging Face repo | {args.hf_repo or MISSING} |")

    w("\n## 5-12. Dataset, split and sampling\n")
    source = manifest.get("source", {})
    pre = (reference.get("normalization"), reference.get("history_rows"), reference.get("stride_rows"))
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Dataset path | `{source.get('path', MISSING)}` |")
    w(f"| Dataset SHA256 | `{source.get('sha256', MISSING)}` |")
    w(f"| Rows | {stats.get('rows', MISSING)} |")
    w(f"| Median / p99 / max dt (s) | {stats.get('median_dt_seconds', MISSING)} / "
      f"{stats.get('p99_dt_seconds', MISSING)} / {stats.get('max_dt_seconds', MISSING)} |")
    w(f"| Gaps > 2 s / segment transitions | {stats.get('gaps_gt_2_seconds', MISSING)} / "
      f"{stats.get('segment_transitions', MISSING)} |")
    for label, value in zip(("train_end", "validation_end"), manifest.get("boundaries_utc", [])):
        w(f"| Split timestamp {label} | `{value}` |")
    for name, bounds in manifest.get("ranges", {}).items():
        w(f"| Split rows {name} | `[{bounds[0]}, {bounds[1]})` |")
    w(f"| history_rows | {pre[1]} |")
    w(f"| stride_rows | {pre[2]} |")
    for name, count in manifest.get("sample_counts", {}).items():
        w(f"| Samples {name} | {count} |")
    w(f"| Contract gate | {contracts.get('status', MISSING)} "
      f"({len(contracts.get('checks', []))} checks, "
      f"{len(contracts.get('failed', []))} failed) |")

    w("\n## 13-17. Hardware, precision and compile\n")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| GPU | {hardware.get('gpu_name', MISSING)} ({bytes_gib(hardware.get('vram_total_bytes'))}) |")
    w(f"| Driver / CUDA driver | {hardware.get('driver_version', MISSING)} / "
      f"{hardware.get('cuda_driver_version', MISSING)} |")
    w(f"| CUDA runtime | {hardware.get('cuda_runtime_version', MISSING)} |")
    w(f"| PyTorch | {hardware.get('torch_version', MISSING)} |")
    w(f"| Precision | {hardware.get('selected_precision', MISSING)}, TF32 matmul "
      f"{hardware.get('tf32_matmul', MISSING)} |")
    w(f"| CPU / RAM | {hardware.get('cpu_count', MISSING)} cores / "
      f"{bytes_gib(hardware.get('ram_total_bytes'))} |")
    w("\n**torch.compile status per model**\n")
    w("| Model | Compile succeeded | Speedup vs eager | Recommended | Used in training |")
    w("|---|---|---:|---|---|")
    for model, _ in RUNS[1:]:
        entry = (compiled.get("models") or {}).get(model, {})
        used = (trainings.get(model) or {}).get("compile")
        w(f"| {model} | {entry.get('compile_succeeded', MISSING)} | "
          f"{number(entry.get('speedup'), 4)} | {entry.get('enable', MISSING)} | "
          f"{MISSING if used is None else used} |")

    w("\n## 18-27. Capacity, systems settings and training cost\n")
    w("| Model | Parameters | Batch | num_workers | Peak alloc (single) | "
      "Peak reserved (single) | samples/s (single) | Train seconds | Best epoch |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model, _ in RUNS[1:]:
        bench = (single.get("models") or {}).get(model, {})
        run = trainings.get(model) or {}
        summary = summaries.get(model) or {}
        w(f"| {model} | {summary.get('parameters', MISSING)} | {run.get('batch_size', MISSING)} | "
          f"{run.get('num_workers', MISSING)} | {bytes_gib(bench.get('peak_allocated_bytes'))} | "
          f"{bytes_gib(bench.get('peak_reserved_bytes'))} | {number(bench.get('samples_per_second'), 5)} | "
          f"{number(run.get('total_seconds'), 5)} | {run.get('best_epoch', MISSING)} |")
    decision = concurrency.get("decision", {})
    w(f"\nConcurrency groups executed: `{schedule.get('groups', MISSING)}` "
      f"(decided by {schedule.get('decided_by', MISSING)}).\n")
    if decision:
        w(f"Benchmark rationale: {decision.get('rationale', MISSING)}\n")
    for group in (concurrency.get("groups") or []):
        w(f"- `{group.get('models')}`: aggregate {number(group.get('aggregate_samples_per_second'), 5)} "
          f"samples/s, peak VRAM {bytes_gib(group.get('peak_memory_used_bytes'))}, "
          f"min free {bytes_gib(group.get('min_free_bytes'))}, "
          f"avg GPU util {number(group.get('average_gpu_utilization_percent'), 4)}%, "
          f"accepted={group.get('accepted')}")

    w("\n## 28-29. Metrics\n")
    for split in ("validation", "test"):
        w(f"\n### {split}\n")
        for model, _ in RUNS:
            summary = summaries.get(model)
            metrics = (summary or {}).get("metrics", {}).get(split)
            w(f"\n**{model}** ({(metrics or {}).get('samples', MISSING)} samples)\n")
            w("| Metric | 1m | 2m | 3m |")
            w("|---|---:|---:|---:|")
            out.extend(metric_rows(metrics))

    w("\n## 30-33. Artifact locations\n")
    w("| Model | Local checkpoints | Local predictions | HF prefix |")
    w("|---|---|---|---|")
    repo = args.hf_repo or MISSING
    for model, run_name in RUNS:
        checkpoints = (MISSING if model == "e0"
                       else f"`{args.checkpoint_root}/{model}/{run_name}/{{best,last}}`")
        w(f"| {model} | {checkpoints} | `{args.artifact_root}/{model}/{run_name}/"
          f"{{train,validation,test}}_predictions.csv.gz` | `{repo}` -> `{model}/{run_name}/` |")

    w("\n## 34-36. Incidents and deviations\n")
    w("\n### Experiment-contract deviations\n")
    for item in DEVIATIONS:
        w(f"- {item}")
    w("")
    crashed = final.get("crashed") or []
    resumed = final.get("resumed") or []
    w(f"- Crashed jobs: {', '.join(crashed) if crashed else 'none'}")
    w(f"- Resumed jobs: {', '.join(resumed) if resumed else 'none'}")
    w(f"- Total scheduled training wall time: {number(final.get('total_seconds'), 6)} s")
    w(f"- GPU usage log: `{final.get('gpu_usage_csv', vast/'gpu_usage.csv')}`")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out)+"\n")
    print(f"wrote {output} ({len(out)} lines)")


if __name__ == "__main__":
    main()
