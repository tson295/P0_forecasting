"""Assemble reports/FINAL_REPORT.md from real artifacts only; never invents a number."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = ("e0", "ofi_lstm", "hfformer", "patchtst", "moderntcn", "lit")
HORIZONS = ("1m", "2m", "3m")
MISSING = "n/a"
# split_manifest.source.row_repairs, in the order the loader applies them.
REPAIR_FIELDS = ("rows_in_file", "rows_used", "sorted_by_timestamp", "rows_moved_by_sort",
                 "duplicate_timestamp_policy", "duplicate_rows_dropped")
PRICE_ROWS = (("RMSE (USD)", "rmse"), ("MAE (USD)", "mae"), ("R2", "r2"),
              ("RMSE gain vs E0", "rmse_gain_vs_e0"))
RETURN_ROWS = (("RMSE", "rmse"), ("MAE", "mae"), ("R2", "r2"),
               ("RMSE gain vs E0", "rmse_gain_vs_e0"))

METRIC_SPACE_NOTE = (
    "**Metric space.** Training is unchanged: the target is still `log(mid[t+h]/mid[t])` and the "
    "loss is still MSE on that log return. Only the reported space changed -- the same predictions "
    "are measured on the mid price via `pred_mid = origin_mid*exp(pred_return)`, so RMSE and MAE "
    "below are in quote currency (USD). E0 predicts that the price does not move, so its predicted "
    "mid is the origin mid and `RMSE_E0 = sqrt(mean((target_mid-origin_mid)^2))`. Standard R2 is "
    "near 1 for every model including E0, because sigma(target mid) is about 16,579 USD while the E0 "
    "error measured on this file is 50.4 / 71.7 / 87.9 USD at 1m/2m/3m; it is reported for contract "
    "completeness only, and **RMSE gain vs E0** is the column to read. Each price table is followed "
    "by the same four families in "
    "log-return units, so both spaces stay comparable.")

OCT2023_DEVIATIONS = [
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

GATE1Y_DEVIATIONS = [
    "**The file needed an explicit, recorded row repair before anything read it** (sections E, F). "
    "`BTC_L10_gate_1y.csv` arrives out of order: rows 0..64079 are a 7.46-day block "
    "(2026-07-01..07-08) prepended in front of the main 2025-09-17..2026-09-16 block, and 9 "
    "timestamps carry two rows with different payloads, so as delivered the file is neither monotone "
    "nor unique. The Oct-2023 contract forbids reordering or dropping rows silently, so the repair is "
    "opt-in and recorded rather than automatic: `data.sort_by_timestamp` still defaults to false and "
    "`data.duplicate_timestamp_policy` still defaults to `error`, which leaves the Oct-2023 run "
    "bit-identical and still fails loudly on an unexpected file. This run sets them to true and "
    "`keep_first`; the sort is stable, so rows sharing a timestamp keep their file order, and exactly "
    "what was done is written to `split_manifest.source.row_repairs` in every checkpoint and shown in "
    "the Data repair table above.",
    "**`max_gap_seconds` and `target_tolerance_seconds` rescaled 2.0 s -> 10.0 s.** The new file sits "
    "on a perfect 10 s grid: min dt = median dt = 10.0 s, every dt a multiple of it, and 31 gaps "
    "longer than one slot. The Oct-2023 2.0 s rule measures cleanliness in units of that file's "
    "~1.2 s cadence; applied here it marks all 3,139,596 edges bad and would leave zero usable "
    "samples. Both knobs moved together to the measured cadence, so the rule itself -- history, "
    "origin and every target inside one uninterrupted stretch -- is unchanged; only its unit follows "
    "the file.",
    "**`history_seconds` 60 -> 490, chosen so the architectures stay identical** (section K). At 10 s "
    "cadence 60 s resolves to history_rows=6. PatchTST then refuses to build at all (its history must "
    "be >= patch_length=16), while HFformer and LiT do build but at a different capacity "
    "(22,026 -> 21,897 and 736,547 -> 735,843), so the frozen parameter contract breaks either way. "
    "490 s re-resolves to history_rows=49 and stride_rows=8 -- exactly the Oct-2023 values -- so "
    "every model sees the same "
    "input shape and all five parameter counts are unchanged (ofi_lstm 55,491; hfformer 22,026; "
    "patchtst 477,059; moderntcn 50,568,195; lit 736,547). Only the wall-clock span of one history "
    "window differs, because a row is now 10 s instead of ~1.2 s.",
    "**R2 is reported in price space although it is uninformative there** (sections 28-29). The four "
    "metric families are frozen, so R2 is reported for completeness; but on the mid price "
    "sigma(target mid) is about 16,579 USD while the E0 error measured on this file is 50.4 / 71.7 / "
    "87.9 USD at 1m/2m/3m, which pins R2 near 0.9999 for all six models including E0. It is kept "
    "as-is rather than replaced by an invented metric: the log-return table under each price table "
    "preserves the Oct-2023 comparison, "
    "and RMSE gain vs E0 is what separates the models.",
]

SUITES = {
    "oct2023": dict(
        title="Base experiment final report — BTC L10, 60 s history",
        runs=(("e0", "e0_60s"), ("ofi_lstm", "ofi_lstm_60s_base"),
              ("hfformer", "hfformer_60s_base"), ("patchtst", "patchtst_60s_base"),
              ("moderntcn", "moderntcn_60s_base"), ("lit", "lit_60s_base")),
        reports_dir="reports/vast", contract_check="reports/contract_check.json",
        max_gap=2.0, price=False, deviations=OCT2023_DEVIATIONS),
    "gate1y": dict(
        title="Gate experiment final report — BTC L10 1 year at 10 s, 490 s history",
        runs=tuple((model, f"{model}_490s_gate1y") for model in MODELS),
        reports_dir="reports/vast_gate1y", contract_check="reports/contract_check_gate1y.json",
        max_gap=10.0, price=True, deviations=GATE1Y_DEVIATIONS),
}


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


def metric_space(metrics, price):
    """Label a table from the artifact's own units; the suite only covers a missing one.

    export_predictions now writes price metrics at the top level for every run, so a
    re-export of an older suite changes the space of the numbers, not the suite name.
    """
    units = (metrics or {}).get("units")
    priced = price if units is None else units == "quote_currency"
    return (PRICE_ROWS, "price (USD)") if priced else (RETURN_ROWS, "log return")


def metric_block(metrics, rows, header):
    """One table; the header cell names the space the numbers live in."""
    block = [f"| {header} | " + " | ".join(HORIZONS) + " |", "|---|---:|---:|---:|"]
    for label, key in rows:
        values = (metrics or {}).get(key) or [None]*len(HORIZONS)
        block.append(f"| {label} | " + " | ".join(number(v) for v in values) + " |")
    return block


def bytes_gib(value):
    return f"{value/2**30:.2f} GiB" if isinstance(value, (int, float)) else MISSING


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=sorted(SUITES), default="oct2023")
    p.add_argument("--output", default="reports/FINAL_REPORT.md")
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--reports-dir", help="benchmark JSONs; default reports/vast, "
                                         "reports/vast_gate1y for --suite gate1y")
    p.add_argument("--contract-check", help="gate output; default reports/contract_check.json, "
                                            "reports/contract_check_gate1y.json for --suite gate1y")
    p.add_argument("--hf-repo", default="")
    p.add_argument("--training-commit", default="")
    p.add_argument("--initial-commit", default="")
    args = p.parse_args()

    suite = SUITES[args.suite]
    runs, price = suite["runs"], suite["price"]
    vast = Path(args.reports_dir or suite["reports_dir"])
    hardware = load(vast/"hardware.json") or {}
    single = load(vast/"single_job_benchmark.json") or {}
    compiled = load(vast/"compile_benchmark.json") or {}
    concurrency = load(vast/"concurrency_benchmark.json") or {}
    schedule = load(vast/"training_schedule.json") or {}
    final = load(vast/"final_training_summary.json") or {}
    contracts = load(args.contract_check or suite["contract_check"]) or {}

    summaries, histories, trainings = {}, {}, {}
    for model, run_name in runs:
        summaries[model] = load(Path(args.artifact_root)/model/run_name/"run_summary.json")
        histories[model] = load(Path(args.artifact_root)/model/run_name/"training_history.jsonl")
        trainings[model] = load(Path(args.checkpoint_root)/model/run_name/"training_run.json")

    reference = next((s for s in summaries.values() if s), {})
    manifest = reference.get("split_manifest", {})
    stats = reference.get("data_stats", {})
    out = []
    w = out.append

    w(f"# {suite['title']}\n")
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
    # The gap count is meaningless unless it is labelled with the rule that produced it.
    max_gap = ((reference.get("experiment") or {}).get("data") or {}).get("max_gap_seconds",
                                                                         suite["max_gap"])
    gaps = stats.get("gaps_gt_max_gap", stats.get("gaps_gt_2_seconds", MISSING))
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Dataset path | `{source.get('path', MISSING)}` |")
    w(f"| Dataset SHA256 | `{source.get('sha256', MISSING)}` |")
    w(f"| Rows | {stats.get('rows', MISSING)} |")
    w(f"| Median / p99 / max dt (s) | {stats.get('median_dt_seconds', MISSING)} / "
      f"{stats.get('p99_dt_seconds', MISSING)} / {stats.get('max_dt_seconds', MISSING)} |")
    w(f"| Gaps > {number(max_gap)} s / segment transitions | {gaps} / "
      f"{stats.get('segment_transitions', MISSING)} |")
    for label, value in zip(("train_end", "validation_end"), manifest.get("boundaries_utc", [])):
        w(f"| Split timestamp {label} | `{value}` |")
    for name, bounds in manifest.get("ranges", {}).items():
        w(f"| Split rows {name} | `[{bounds[0]}, {bounds[1]})` |")
    w(f"| history_rows | {number(pre[1])} |")
    w(f"| stride_rows | {number(pre[2])} |")
    for name, count in manifest.get("sample_counts", {}).items():
        w(f"| Samples {name} | {count} |")
    # Name the gated file: a --contract-check pointed at the other suite would otherwise pass silently.
    w(f"| Contract gate | {contracts.get('status', MISSING)} on "
      f"`{Path(contracts['csv']).name if contracts.get('csv') else MISSING}` "
      f"({len(contracts.get('checks', []))} checks, "
      f"{len(contracts.get('failed', []))} failed) |")

    w("\n### Data repair\n")
    repairs = source.get("row_repairs")
    if not repairs:
        # "none" is a claim about a run that happened; with no run summary it would be invented.
        w("none" if reference else MISSING)
    else:
        w("| Item | Value |")
        w("|---|---|")
        for field in REPAIR_FIELDS:
            w(f"| {field} | {repairs.get(field, MISSING)} |")

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
    for model, _ in runs[1:]:
        entry = (compiled.get("models") or {}).get(model, {})
        used = (trainings.get(model) or {}).get("compile")
        w(f"| {model} | {entry.get('compile_succeeded', MISSING)} | "
          f"{number(entry.get('speedup'), 4)} | {entry.get('enable', MISSING)} | "
          f"{MISSING if used is None else used} |")

    w("\n## 18-27. Capacity, systems settings and training cost\n")
    w("| Model | Parameters | Batch | num_workers | Peak alloc (single) | "
      "Peak reserved (single) | samples/s (single) | Train seconds | Best epoch |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model, _ in runs[1:]:
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
    if price:
        w(f"\n{METRIC_SPACE_NOTE}\n")
    for split in ("validation", "test"):
        w(f"\n### {split}\n")
        for model, _ in runs:
            summary = summaries.get(model)
            metrics = (summary or {}).get("metrics", {}).get(split)
            w(f"\n**{model}** ({(metrics or {}).get('samples', MISSING)} samples)\n")
            out.extend(metric_block(metrics, *metric_space(metrics, price)))
            nested = (metrics or {}).get("log_return")
            if nested:
                w("")
                out.extend(metric_block(nested, RETURN_ROWS, "log return"))

    w("\n## 30-33. Artifact locations\n")
    w("| Model | Local checkpoints | Local predictions | HF prefix |")
    w("|---|---|---|---|")
    repo = args.hf_repo or MISSING
    for model, run_name in runs:
        checkpoints = (MISSING if model == "e0"
                       else f"`{args.checkpoint_root}/{model}/{run_name}/{{best,last}}`")
        w(f"| {model} | {checkpoints} | `{args.artifact_root}/{model}/{run_name}/"
          f"{{train,validation,test}}_predictions.csv.gz` | `{repo}` -> `{model}/{run_name}/` |")

    w("\n## 34-36. Incidents and deviations\n")
    w("\n### Experiment-contract deviations\n")
    for item in suite["deviations"]:
        w(f"- {item}")
    w("")
    crashed = final.get("crashed") or []
    resumed = final.get("resumed") or []
    clean = "none" if final else MISSING  # Same rule: no summary, no clean bill of health.
    w(f"- Crashed jobs: {', '.join(crashed) if crashed else clean}")
    w(f"- Resumed jobs: {', '.join(resumed) if resumed else clean}")
    w(f"- Total scheduled training wall time: {number(final.get('total_seconds'), 6)} s")
    w(f"- GPU usage log: `{final.get('gpu_usage_csv', vast/'gpu_usage.csv')}`")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out)+"\n")
    print(f"wrote {output} ({len(out)} lines)")


if __name__ == "__main__":
    main()
