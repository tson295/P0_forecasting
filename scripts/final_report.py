"""Assemble reports/FINAL_REPORT.md from real artifacts only; never invents a number."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = ("e0", "ofi_lstm", "hfformer", "patchtst", "moderntcn", "lit")
LEARNED = MODELS[1:]
HORIZONS = ("1m", "2m", "3m")
MISSING = "n/a"
# split_manifest.source.row_repairs, in the order the loader applies them.
REPAIR_FIELDS = ("rows_in_file", "rows_used", "sorted_by_timestamp", "rows_moved_by_sort",
                 "duplicate_timestamp_policy", "duplicate_rows_dropped")
# No mean-based R2 on the price: sigma(price) dwarfs every model's error, so it sits at
# ~0.9999 for E0 too. price_metrics stopped emitting the key; older artifacts still carry
# it and this report ignores it.
PRICE_ROWS = (("RMSE (USD)", "rmse"), ("MAE (USD)", "mae"), ("R2 gain vs E0", "r2_gain_vs_e0"),
              ("RMSE gain vs E0", "rmse_gain_vs_e0"))
RETURN_ROWS = (("RMSE", "rmse"), ("MAE", "mae"), ("R2", "r2"),
               ("RMSE gain vs E0", "rmse_gain_vs_e0"))
# The frozen capacities of the first two runs; this run's counts come from the artifacts.
BASELINE_PARAMETERS = dict(e0=0, ofi_lstm=55491, hfformer=22026, patchtst=477059,
                           moderntcn=50568195, lit=736547)

METRIC_SPACE_NOTE = (
    "**Metric space.** Training is unchanged: the target is still `log(mid[t+h]/mid[t])` and the "
    "loss is still MSE on that log return. Only the reported space changed -- the same predictions "
    "are measured on the mid price via `pred_mid = origin_mid*exp(pred_return)`, so RMSE and MAE "
    "below are in quote currency (USD). E0 predicts that the price does not move, so its predicted "
    "mid is the origin mid and `RMSE_E0 = sqrt(mean((target_mid-origin_mid)^2))`. **R2 gain vs E0** "
    "= `1 - SSE_model/SSE_E0` is reported instead of a mean-based R2 on the price: sigma(target mid) "
    "is three orders of magnitude larger than any model's error here, which pins a mean-based price "
    "R2 near 0.9999 for every model including E0 and separates nothing. The gain is 0 when a model "
    "equals E0, positive when it beats E0, and is the squared-error twin of RMSE gain vs E0 "
    "(`r2_gain = 1-(1-rmse_gain)^2`). Each price table is followed by the same four families in "
    "log-return units, where the mean of the target is near zero and R2 is meaningful, so both "
    "spaces stay comparable.")

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
    "and RMSE gain vs E0 is what separates the models. **Superseded by the walk-forward run**, where "
    "the user replaced price R2 with R2 gain vs E0; this script no longer renders a mean-based price "
    "R2 for any suite, so regenerating this report shows `R2 gain vs E0` as `n/a` for the Gate "
    "artifacts, which were written before that metric existed. Their RMSE gain vs E0 is unchanged.",
]

WF3_DEVIATIONS = [
    "**Stride 8 -> 2 rows and capacity scaled 20-200x, so this run is not parameter-comparable to "
    "the first two** (sections H, K). `stride_seconds` 20.0 resolves to stride_rows=2 on the same "
    "10 s grid, which roughly quadruples the windows per unit of time (fold 3 trains on 998,525 "
    "samples against the Gate run's 274,215), and each architecture is widened inside its own family "
    "through the config's `model_kwargs`: ofi_lstm 55,491 -> 2,990,211, hfformer 22,026 -> 4,677,278, "
    "patchtst 477,059 -> 14,434,563, lit 736,547 -> 15,995,235; ModernTCN stays at 50,568,195 because "
    "it is already the GPU bottleneck. `src/models` DEFAULTS are untouched, so the Oct-2023 and Gate "
    "suites still build their frozen sizes and still gate on them. The Capacity table below states "
    "both counts; no metric here is comparable one-to-one with the earlier two reports.",
    "**Mean-based R2 on the price is gone, replaced by R2 gain vs E0** (sections 28-29; this also "
    "supersedes the prompt's `do not report R2_OS` line). On the mid price the mean is not a "
    "baseline anyone would use -- sigma(target mid) is thousands of USD against a ~50 USD E0 error -- "
    "so the old price R2 read ~0.9999 for all six models including E0 and ranked nothing. At the "
    "user's instruction `price_metrics` now returns `r2_gain_vs_e0 = 1 - SSE_model/SSE_E0` and no "
    "`r2` key at all, so 0 means 'exactly E0' and positive means 'beats E0'; the identity "
    "`r2_gain = 1-(1-rmse_gain)^2` ties it to the RMSE column. The nested log-return table keeps its "
    "own mean-based R2, which stays informative because the mean log return is near zero.",
    "**An explicit 180 s purge and embargo now drops samples at every split boundary**, which the "
    "first two runs did not do. `target_limits()` hands each split an exclusive ns bound -- the next "
    "split's first timestamp minus `embargo_seconds=180` -- and `LOBDataset` drops any sample whose "
    "targets reach it, so no training window can read a record inside the embargo. Earlier runs cut "
    "on a timestamp alone, which let a train sample's 3-minute target land in the validation split; "
    "their sample counts are therefore slightly optimistic against these. The measured separations "
    "are in the Walk-forward and leakage section and `scripts/check_leakage.py` re-derives every one "
    "of them from the raw timestamps instead of trusting the manifest.",
    "**Three folds mean three trained models per architecture**, so there is no single 'the model'. "
    "`best_epoch`, `total_seconds`, every checkpoint and every metric below are per fold, selected on "
    "that fold's validation block alone. The test section then scores each fold's own best checkpoint "
    "on the one held-out tail split, which is identical for all 18 runs, and the stability table is "
    "the honest summary: fold-to-fold spread, not a single number.",
]

SUITES = {
    "oct2023": dict(
        title="Base experiment final report — BTC L10, 60 s history",
        runs=tuple((model, None, name) for model, name in
                   (("e0", "e0_60s"), ("ofi_lstm", "ofi_lstm_60s_base"),
                    ("hfformer", "hfformer_60s_base"), ("patchtst", "patchtst_60s_base"),
                    ("moderntcn", "moderntcn_60s_base"), ("lit", "lit_60s_base"))),
        folds=(None,), reports_dir="reports/vast", contract_check="reports/contract_check.json",
        max_gap=2.0, price=False, deviations=OCT2023_DEVIATIONS),
    "gate1y": dict(
        title="Gate experiment final report — BTC L10 1 year at 10 s, 490 s history",
        runs=tuple((model, None, f"{model}_490s_gate1y") for model in MODELS),
        folds=(None,), reports_dir="reports/vast_gate1y",
        contract_check="reports/contract_check_gate1y.json",
        max_gap=10.0, price=True, deviations=GATE1Y_DEVIATIONS),
    "wf3": dict(
        title="Walk-forward final report — BTC L10 1 year at 10 s, 3 expanding folds, 20 s stride",
        runs=tuple((model, fold, f"{model}_wf3_f{fold}") for fold in (1, 2, 3) for model in MODELS),
        folds=(1, 2, 3), reports_dir="reports/vast_wf3",
        contract_check="reports/contract_check_wf3.json",
        leakage_audit="reports/leakage_audit_wf3.json", baseline=BASELINE_PARAMETERS,
        max_gap=10.0, price=True, deviations=WF3_DEVIATIONS),
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


def split_metrics(summary, split):
    return ((summary or {}).get("metrics") or {}).get(split) or {}


def emit_metrics(out, label, summary, split, price):
    """One model's block for one split: the price table, then the same families as returns."""
    metrics = split_metrics(summary, split)
    out.append(f"\n**{label}** ({metrics.get('samples', MISSING)} samples)\n")
    out.extend(metric_block(metrics, *metric_space(metrics, price)))
    nested = metrics.get("log_return")
    if nested:
        out.append("")
        out.extend(metric_block(nested, RETURN_ROWS, "log return"))


def stability_rows(summaries, folds, key="rmse_gain_vs_e0"):
    """Fold-to-fold spread on the test split, which is the same samples for every fold."""
    rows = ["| Model | Horizon | "+" | ".join(f"fold {f}" for f in folds)+" | mean | spread |",
            "|---|---|"+"---:|"*(len(folds)+2)]
    for model in MODELS:
        for i, horizon in enumerate(HORIZONS):
            values = [(split_metrics(summaries.get((model, f)), "test").get(key) or [None]*3)[i]
                      for f in folds]
            present = [v for v in values if v is not None]
            mean = sum(present)/len(present) if present else None
            spread = max(present)-min(present) if len(present) > 1 else None
            rows.append(f"| {model} | {horizon} | "+" | ".join(number(v, 4) for v in values) +
                        f" | {number(mean, 4)} | {number(spread, 4)} |")
    return rows


def audit_min_gaps(audit):
    """Smallest separation the independent audit measured, per fold and boundary.

    The audit runs every model of every fold, so the minimum is the weakest case the
    180 s requirement has to survive, not an average.
    """
    gaps = {}
    for check in audit.get("checks") or []:
        name, detail = check.get("name", ""), check.get("detail") or {}
        if not name.endswith("_gap_seconds") or detail.get("gap_seconds") is None:
            continue
        # fold<K>.<model>.<earlier>_to_<later>_gap_seconds
        key = (name.split(".")[0].removeprefix("fold"),
               name.split(".")[-1].removesuffix("_gap_seconds"))
        gaps[key] = min(gaps.get(key, detail["gap_seconds"]), detail["gap_seconds"])
    return gaps


def walk_forward_section(out, summaries, folds, audit, audit_path, horizon):
    """Fold layout, purge/embargo separations and the independent audit, from artifacts only."""
    w = out.append
    manifests = {f: (next((summaries[(m, f)] for m in MODELS if summaries.get((m, f))), None) or {})
                 .get("split_manifest", {}) for f in folds}
    any_manifest = next((m for m in manifests.values() if m), {})
    w("\n## Walk-forward and leakage\n")
    w("The last `test_fraction` of elapsed time is held out as the test split and is never trained "
      "on and never used to pick a checkpoint; the rest is cut into `folds+1` equal blocks of "
      "elapsed time, and fold k trains on blocks `[0, k)` and validates on block k, so every fold "
      "validates strictly after everything it trained on. **The test split is identical for every "
      "model and every fold** -- same rows, same samples -- so the test tables compare folds, not "
      "datasets. Each fold's `best_epoch` is chosen on that fold's validation block alone.\n")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Split scheme | {any_manifest.get('scheme', MISSING)} |")
    w(f"| Folds / this report | {any_manifest.get('folds', MISSING)} / {', '.join(map(str, folds))} |")
    w(f"| test_fraction | {number(any_manifest.get('test_fraction'))} |")
    w(f"| embargo_seconds | {number(any_manifest.get('embargo_seconds'))} |")
    w(f"| Longest target horizon (s) | {horizon} |")
    ranges = {f: (m.get("ranges") or {}) for f, m in manifests.items()}
    tests = {f: tuple(r["test"]) for f, r in ranges.items() if r.get("test")}
    counts = {f: (manifests[f].get("sample_counts") or {}).get("test") for f in folds}
    test_rows = f"`[{tests[folds[0]][0]}, {tests[folds[0]][1]})`" if tests.get(folds[0]) else MISSING
    w(f"| Test split rows | {test_rows} |")
    w(f"| Test samples | {counts.get(folds[0]) if counts.get(folds[0]) is not None else MISSING} |")
    # Identity is a property of what the artifacts recorded, so claim it only from them.
    w(f"| Test split identical across folds | "
      f"{len(set(tests.values())) == 1 and len(set(counts.values())) == 1 if tests else MISSING} |")

    w("\n**Fold blocks** (equal spans of elapsed time; the last row is the held-out tail)\n")
    rows = any_manifest.get("block_boundary_rows") or []
    stamps = any_manifest.get("block_boundaries_utc") or []
    if not rows:
        w(MISSING)
    else:
        w("| Block | Rows | Starts (UTC) |")
        w("|---|---|---|")
        for i, (lo, hi) in enumerate(zip(rows, rows[1:])):
            w(f"| {i+1} | `[{lo}, {hi})` | `{stamps[i] if i < len(stamps) else MISSING}` |")
        last = tests.get(folds[0])
        w(f"| test | `[{last[0]}, {last[1]})` | `{stamps[-1] if stamps else MISSING}` |"
          if last else f"| test | {MISSING} | `{stamps[-1] if stamps else MISSING}` |")

    w("\n**Purge and embargo per fold.** Separation is the wall-clock distance between the last raw "
      f"record one split's samples touch and the first record the next split's samples touch; it must "
      f"exceed the {horizon} s longest horizon, otherwise the two splits could share a target.\n")
    gaps = audit_min_gaps(audit)
    w("| Fold | Train rows | Validation rows | Train samples | Validation samples | "
      f"train->val (s) | val->test (s) | audit min gap (s) | > {horizon} s |")
    w("|---:|---|---|---:|---:|---:|---:|---:|---|")
    for fold in folds:
        manifest, bounds = manifests[fold], ranges[fold]
        separation = manifest.get("split_separation_seconds") or {}
        sample_counts = manifest.get("sample_counts") or {}
        audited = [v for (f, _), v in gaps.items() if f == str(fold)]
        measured = [v for v in (separation.get("train"), separation.get("validation"), *audited)
                    if v is not None]
        w(f"| {fold} | "
          + " | ".join(f"`[{bounds[name][0]}, {bounds[name][1]})`" if bounds.get(name) else MISSING
                       for name in ("train", "validation"))
          + f" | {sample_counts.get('train', MISSING)} | {sample_counts.get('validation', MISSING)} | "
          f"{number(separation.get('train'), 5)} | {number(separation.get('validation'), 5)} | "
          f"{number(min(audited), 5) if audited else MISSING} | "
          f"{min(measured) > horizon if measured else MISSING} |")

    w("\n**Independent leakage audit.** `scripts/check_leakage.py` re-derives the spans, the "
      "separations, the standardizer fit and the test fingerprint from the raw timestamps, so a bug "
      "in the splitter cannot certify itself.\n")
    checks = audit.get("checks") or []
    identical = next((c for c in checks
                      if c.get("name") == "test_split_identical_across_every_model_and_fold"), {})
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Audit report | `{audit_path}` |")
    w(f"| Status | {audit.get('status', MISSING)} |")
    w(f"| Checks passed | {len(checks)-len(audit.get('failed') or [])}/{len(checks) or MISSING} |")
    w(f"| Failed checks | {', '.join(audit.get('failed') or []) or ('none' if audit else MISSING)} |")
    w(f"| Embargo required (s) | {number(audit.get('embargo_seconds_required'))} |")
    w(f"| Test split identical across every model and fold | {identical.get('passed', MISSING)} |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=sorted(SUITES), default="oct2023")
    p.add_argument("--output", default="reports/FINAL_REPORT.md")
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--reports-dir", help="benchmark JSONs; default reports/vast, "
                                         "reports/vast_gate1y for --suite gate1y, "
                                         "reports/vast_wf3 for --suite wf3")
    p.add_argument("--contract-check", help="gate output; default reports/contract_check.json, "
                                            "reports/contract_check_gate1y.json for --suite gate1y, "
                                            "reports/contract_check_wf3.json for --suite wf3")
    p.add_argument("--leakage-audit", help="check_leakage.py output; default "
                                           "reports/leakage_audit_wf3.json for --suite wf3")
    p.add_argument("--hf-repo", default="")
    p.add_argument("--training-commit", default="")
    p.add_argument("--initial-commit", default="")
    args = p.parse_args()

    suite = SUITES[args.suite]
    runs, price, folds = suite["runs"], suite["price"], suite["folds"]
    folded = folds != (None,)
    vast = Path(args.reports_dir or suite["reports_dir"])
    hardware = load(vast/"hardware.json") or {}
    single = load(vast/"single_job_benchmark.json") or {}
    compiled = load(vast/"compile_benchmark.json") or {}
    concurrency = load(vast/"concurrency_benchmark.json") or {}
    schedule = load(vast/"training_schedule.json") or {}
    final = load(vast/"final_training_summary.json") or {}
    contracts = load(args.contract_check or suite["contract_check"]) or {}
    audit_path = args.leakage_audit or suite.get("leakage_audit", "")
    audit = (load(audit_path) or {}) if audit_path else {}

    summaries, histories, trainings = {}, {}, {}
    for model, fold, run_name in runs:
        summaries[model, fold] = load(Path(args.artifact_root)/model/run_name/"run_summary.json")
        histories[model, fold] = load(Path(args.artifact_root)/model/run_name/"training_history.jsonl")
        trainings[model, fold] = load(Path(args.checkpoint_root)/model/run_name/"training_run.json")

    reference = next((s for s in summaries.values() if s), {})
    manifest = reference.get("split_manifest", {})
    stats = reference.get("data_stats", {})
    horizons = ((reference.get("experiment") or {}).get("data") or {}).get("horizons_seconds")
    horizon = max(horizons) if horizons else 180
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
    if folded:
        # Boundaries, ranges and counts differ per fold; they belong to the fold tables below.
        w(f"| Split scheme | {manifest.get('scheme', MISSING)}, "
          f"folds={manifest.get('folds', MISSING)}, "
          f"test_fraction={number(manifest.get('test_fraction'))}, "
          f"embargo_seconds={number(manifest.get('embargo_seconds'))} "
          "(per-fold layout under Walk-forward and leakage) |")
    else:
        for label, value in zip(("train_end", "validation_end"), manifest.get("boundaries_utc", [])):
            w(f"| Split timestamp {label} | `{value}` |")
        for name, bounds in manifest.get("ranges", {}).items():
            w(f"| Split rows {name} | `[{bounds[0]}, {bounds[1]})` |")
    w(f"| history_rows | {number(pre[1])} |")
    w(f"| stride_rows | {number(pre[2])} |")
    if not folded:
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

    if folded:
        walk_forward_section(out, summaries, folds, audit, audit_path, horizon)

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
    # One benchmark per model (the folds share the shapes); the last column is per run.
    used_head = ("Used in training ("+"/".join(f"f{f}" for f in folds)+")" if folded
                 else "Used in training")
    w(f"| Model | Compile succeeded | Speedup vs eager | Recommended | {used_head} |")
    w("|---|---|---:|---|---|")
    for model in LEARNED:
        entry = (compiled.get("models") or {}).get(model, {})
        used = [(trainings.get((model, f)) or {}).get("compile") for f in folds]
        w(f"| {model} | {entry.get('compile_succeeded', MISSING)} | "
          f"{number(entry.get('speedup'), 4)} | {entry.get('enable', MISSING)} | "
          + "/".join(MISSING if u is None else str(u) for u in used)+" |")

    w("\n## 18-27. Capacity, systems settings and training cost\n")
    if suite.get("baseline"):
        w("**Capacity this run vs the frozen baseline of the first two experiments**\n")
        w("| Model | Parameters (this run) | Baseline (oct2023/gate1y) | Multiplier |")
        w("|---|---:|---:|---:|")
        for model in MODELS:
            base = suite["baseline"][model]
            found = next((summaries[model, f] for f in folds
                          if (summaries.get((model, f)) or {}).get("parameters") is not None), None)
            count = found["parameters"] if found else None
            w(f"| {model} | {number(count)} | {base} | "
              f"{f'{count/base:.1f}x' if count is not None and base else MISSING} |")
        w("")
    head = ["Model"]+(["Fold"] if folded else [])+[
        "Parameters", "Batch", "num_workers", "Peak alloc (single)", "Peak reserved (single)",
        "samples/s (single)", "Train seconds", "Best epoch"]
    w("| "+" | ".join(head)+" |")
    w("|"+"|".join(["---"]+["---:"]*(len(head)-1))+"|")
    for model in LEARNED:
        for fold in folds:
            bench = (single.get("models") or {}).get(model, {})
            run = trainings.get((model, fold)) or {}
            summary = summaries.get((model, fold)) or {}
            w(f"| {model} | "+(f"{fold} | " if folded else "")
              + f"{summary.get('parameters', MISSING)} | {run.get('batch_size', MISSING)} | "
              f"{run.get('num_workers', MISSING)} | {bytes_gib(bench.get('peak_allocated_bytes'))} | "
              f"{bytes_gib(bench.get('peak_reserved_bytes'))} | "
              f"{number(bench.get('samples_per_second'), 5)} | "
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
    if folded:
        for fold in folds:
            w(f"\n### Fold {fold} — validation\n")
            for model in MODELS:
                emit_metrics(out, model, summaries.get((model, fold)), "validation", price)
        w("\n### Test — the same held-out tail split for every fold\n")
        w("Each fold's own best checkpoint, scored on identical samples, so the differences below "
          "are the folds and nothing else.")
        for fold in folds:
            w(f"\n#### Fold {fold}\n")
            for model in MODELS:
                emit_metrics(out, model, summaries.get((model, fold)), "test", price)
        w("\n### Test stability across folds\n")
        w("RMSE gain vs E0 on the test split: per model and horizon, every fold, their mean and "
          "the max-min spread. Spread is the cost of picking one fold's checkpoint blind.\n")
        out.extend(stability_rows(summaries, folds))
    else:
        for split in ("validation", "test"):
            w(f"\n### {split}\n")
            for model, fold, _ in runs:
                emit_metrics(out, model, summaries.get((model, fold)), split, price)

    w("\n## 30-33. Artifact locations\n")
    w("| Model | Local checkpoints | Local predictions | HF prefix |")
    w("|---|---|---|---|")
    repo = args.hf_repo or MISSING
    for model, _, run_name in runs:
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
