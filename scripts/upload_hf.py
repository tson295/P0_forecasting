"""Publish frozen checkpoints, prediction artifacts and a generated model card to a private HF repo."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import filter_repo_objects

from src.models import build_model
from src.training.checkpoint import METADATA_FILES
from src.utils.metrics import HORIZON_LABELS

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
CHECKPOINT_FILES = (("model.safetensors", "config.json", "environment.json",
                     "optimizer.pt", "scheduler.pt", "trainer_state.pt")
                    + tuple(f"{name}.json" for name in METADATA_FILES))
ARTIFACT_FILES = (tuple(f"{s}_predictions.csv.gz" for s in SPLITS)
                  + tuple(f"{s}_metrics.json" for s in SPLITS) + ("run_summary.json",))
# The raw L10 book CSVs never leave the machine; only gzipped prediction tables ship.
IGNORE = ("*.csv", "*.pyc", "*__pycache__/*", "*.venv/*", "*.previous/*", ".*", "*/.*")
DATASET_NAMES = ("BTCUSDT_L10_oct2023.csv", "BTC_L10_gate_1y.csv")
TAGS = ("time-series-forecasting", "limit-order-book", "market-microstructure", "bitcoin", "btcusdt")
R2_LABELS = {"r2": "R2", "r2_gain_vs_e0": "R2 gain vs E0"}
OPTIMIZER = "AdamW(lr, weight_decay) + CosineAnnealingLR(T_max=epochs), no warmup, no scheduler tuning"
E0_NOTE = ("Untrained reference baseline: predicts a zero log return at every horizon, so "
           "RMSE_E0 = sqrt(mean(y^2)) and every gain below is measured against it.")
E0_PRICE_NOTE = ("Untrained reference baseline: predicts a zero log return at every horizon, i.e. "
                 "that the mid price does not move, so its predicted mid is the origin mid, "
                 "RMSE_E0 = sqrt(mean((target_mid - origin_mid)^2)) in USD, and every gain below is "
                 "measured against it.")
# The gate 1-year capture, published by two suites: its rows are sorted at load time, so the file's
# first and last line are not the interval.
BOOK_INTERVAL = ("2025-09-17T00:00:00+00:00", "2026-09-16T23:59:50+00:00")
BOOK_REPAIRS = ("rows sorted by timestamp (a 64,080-row 2026-07-01..07-08 block arrives in front "
                "of the main block) and 9 duplicate timestamps carrying different payloads dropped "
                "under `keep_first`: 3,139,606 -> 3,139,597 rows")
# --suite selects one frozen experiment: its run names, its destination repo, the reports directory
# the schedule prose quotes, the frozen contract that stands in until the first checkpoint exists,
# and the prose no metadata file carries.
SUITES = {
    "oct2023": dict(
        repo="{user}/Pretrain_Model", e0=(("e0", "e0_60s"),), reports="reports/vast",
        runs=(("ofi_lstm", "ofi_lstm_60s_base"), ("hfformer", "hfformer_60s_base"),
              ("patchtst", "patchtst_60s_base"), ("moderntcn", "moderntcn_60s_base"),
              ("lit", "lit_60s_base")),
        csv="BTCUSDT_L10_oct2023.csv", contract=None, e0_note=E0_NOTE, sections=(),
        price_metrics=False, r2_key="r2",
        title="BTCUSDT L10 multi-horizon log-return forecasting (60s history)",
        intro="Five learned models plus the untrained `e0` zero-return baseline, trained once under a "
              "frozen experiment contract on a single October 2023 BTCUSDT L10 limit-order-book file. "
              "Every checkpoint (`best`, `last`), every prediction table and every metric file below "
              "was produced by that single run; nothing here is tuned, re-fitted or re-scored."),
    "gate1y": dict(
        repo="Tson29/Pretrain_Model_Gate_1Y", e0=(("e0", "e0_490s_gate1y"),),
        reports="reports/vast_gate1y",
        runs=(("ofi_lstm", "ofi_lstm_490s_gate1y"), ("hfformer", "hfformer_490s_gate1y"),
              ("patchtst", "patchtst_490s_gate1y"), ("moderntcn", "moderntcn_490s_gate1y"),
              ("lit", "lit_490s_gate1y")),
        csv="BTC_L10_gate_1y.csv", contract="contracts/gate1y.json", price_metrics=True,
        r2_key="r2", interval=BOOK_INTERVAL, row_repairs=BOOK_REPAIRS, e0_note=E0_PRICE_NOTE,
        title="BTC L10 multi-horizon forecasting on a one-year 10s book (490s history, raw-price metrics)",
        intro="Five learned models plus the untrained `e0` zero-return baseline, trained once under a "
              "frozen experiment contract on a one-year BTC L10 limit-order-book file sampled on an "
              "exact 10-second grid. Every checkpoint (`best`, `last`), every prediction table and "
              "every metric file below was produced by that single run; nothing here is tuned, "
              "re-fitted or re-scored. The headline metrics are measured in raw price (USD); the "
              "training target is unchanged and still the log return.",
        sections=(
            ("Metric space: raw price, not log return",
             "Training is unchanged from the 60s run: the target is still `log(mid[t+h]/mid[t])` and "
             "the loss is still MSE on that log return. Only the reported metric space changed. Every "
             "`*_metrics.json` here carries RMSE, MAE, standard R2, RMSE_E0 and RMSE gain vs E0 "
             "measured on the mid price in quote currency (USD) at the top level, with the previous "
             "log-return metrics nested under `log_return` so the two runs stay comparable. Every "
             "headline number is recomputed from the exported prediction columns by "
             "`src/utils/predictions.frame_metrics`.\n\n"
             "E0 predicts that the price does not move, so its predicted mid is the origin mid and "
             "`rmse_e0 = sqrt(mean((target_mid - origin_mid)^2))`; measured on this file that baseline "
             "is about 50.4 / 71.7 / 87.9 USD at 1m / 2m / 3m.\n\n"
             "Standard R2 in price space is about 0.9999 for every model, E0 included: sigma(target "
             "mid) is roughly 16,579 USD across the year while every forecast error here is around 50 "
             "USD, so the level of the price, not forecast skill, fills the variance. It is reported "
             "for contract completeness only - `rmse_gain_vs_e0` is the informative column."),
            ("Dataset and what changed against the 60s Oct-2023 run",
             "- A new one-year L10 capture on an exact 10-second grid (SHA256, rows and interval in "
             "the contract table above), 3,139,597 rows after repair, 31 gaps longer than 10s.\n"
             "- The file arrives out of order, so `data.sort_by_timestamp=true` and "
             "`data.duplicate_timestamp_policy=\"keep_first\"` repair it. Both are opt-in and both are "
             "recorded in `split_manifest.source.row_repairs` of every checkpoint; the defaults still "
             "refuse to reorder or deduplicate a book silently.\n"
             "- `max_gap_seconds` and `target_tolerance_seconds` move 2.0s -> 10.0s: at a 10s cadence "
             "the old 2.0s rule marks every edge in the file as a gap.\n"
             "- `history_seconds` moves 60 -> 490, which re-resolves to the same 49 history rows and "
             "the same stride of 8 rows, so the architectures and their parameter counts are "
             "untouched: {parameters}."),
        )),
}
E0_RUN, RUNS = SUITES["oct2023"]["e0"], SUITES["oct2023"]["runs"]


def select_suite(name):
    """Rebind the published run names to one experiment."""
    suite = SUITES[name]
    globals()["E0_RUN"], globals()["RUNS"] = suite["e0"], suite["runs"]
    return suite


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def get(obj, *path, default=None):
    for key in path:
        if not isinstance(obj, dict) or obj.get(key) is None:
            return default
        obj = obj[key]
    return obj


def cell(value, spec="", suffix=""):
    return "n/a" if value is None else (format(value, spec) if spec else str(value))+suffix


def yes_no(value):
    return "n/a" if value is None else ("yes" if value else "no")


def human(size):
    return f"{size/1024**2:.1f} MiB" if size >= 1024**2 else f"{size/1024:.1f} KiB"


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def scan(folder, repo_dir, message, ignore):
    """Preview exactly what upload_folder would send, using the Hub's own pattern filter."""
    found = (sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
             if folder.is_dir() else [])
    files = sorted(filter_repo_objects(found, ignore_patterns=list(ignore)))
    for name in files:
        # Last line of defence: only small monitoring CSVs inside reports/ may ever ship.
        if Path(name).name in DATASET_NAMES or (name.endswith(".csv") and (
                not repo_dir.startswith("reports") or (folder/name).stat().st_size >= 64*1024**2)):
            raise RuntimeError(f"Refusing to upload CSV {folder/name}")
    return dict(local=folder, repo_dir=repo_dir, message=message, files=files, ignore=list(ignore),
                skipped=sorted(set(found)-set(files)),
                bytes=sum((folder/name).stat().st_size for name in files))


def build_plan(monitoring_csv=False):
    # The gpu_usage.csv sampling logs are the only CSVs that may ship, and only on request.
    reports_ignore = (tuple(p for p in IGNORE if p != "*.csv")
                      + tuple(f"*{name}" for name in DATASET_NAMES)) if monitoring_csv else IGNORE
    plan = [scan(ROOT/"artifacts"/E0_RUN[0]/E0_RUN[1], f"{E0_RUN[0]}/{E0_RUN[1]}/artifacts",
                 "e0 baseline artifacts", IGNORE)]
    for model, run in RUNS:
        plan.append(scan(ROOT/"checkpoints"/model/run, f"{model}/{run}", f"{model} checkpoints ({run})", IGNORE))
        plan.append(scan(ROOT/"artifacts"/model/run, f"{model}/{run}/artifacts",
                         f"{model} artifacts ({run})", IGNORE))
    plan.append(scan(ROOT/"reports", "reports", "reports", reports_ignore))
    return plan


def required_paths():
    paths = {f"{E0_RUN[0]}/{E0_RUN[1]}/artifacts/{name}" for name in ARTIFACT_FILES}
    for model, run in RUNS:
        paths |= {f"{model}/{run}/{folder}/{name}"
                  for folder in ("best", "last") for name in CHECKPOINT_FILES}
        paths |= {f"{model}/{run}/artifacts/{name}" for name in ARTIFACT_FILES}
        paths |= {f"{model}/{run}/artifacts/training_history.jsonl", f"{model}/{run}/training_run.json"}
    return paths


def missing_required(plan):
    present = {f"{item['repo_dir']}/{name}" for item in plan for name in item["files"]}
    return sorted(required_paths()-present)


def gather(model, run):
    """Checkpoint metadata is authoritative; run_summary.json carries the same fields for e0, which has none."""
    best = ROOT/"checkpoints"/model/run/"best"
    artifacts = ROOT/"artifacts"/model/run
    summary = load(artifacts/"run_summary.json") or {}
    gathered = (dict(model=model, run=run, summary=summary or None,
                     config=load(best/"config.json") or summary.get("architecture"),
                     environment=load(best/"environment.json"),
                     training=load(ROOT/"checkpoints"/model/run/"training_run.json") or summary.get("training"),
                     metrics={split: load(artifacts/f"{split}_metrics.json") for split in SPLITS})
                | {name: load(best/f"{name}.json") or summary.get(name) for name in METADATA_FILES})
    if not gathered["preprocessing"] and summary:
        gathered["preprocessing"] = {key: summary.get(key)
                                     for key in ("history_rows", "stride_rows", "normalization")}
    return gathered


def contract(runs):
    """Split/preprocessing/target metadata is identical across runs, so take the most complete copy:
    e0 has no checkpoint and only carries a three-key preprocessing stub rebuilt from run_summary."""
    shared = {key: max((run[key] for run in runs if run.get(key)), default=None,
                       key=lambda value: len(value) if isinstance(value, dict) else 0)
              for key in ("experiment", "preprocessing", "target_config", "split_manifest",
                          "data_stats", "environment")}
    if not shared["experiment"]:
        frozen = (load(ROOT/"configs"/f"{run}.json") for _, run in RUNS)
        shared["experiment"] = next((config for config in frozen if config), None)
    return shared


def dataset_interval(manifest, preprocessing):
    """Book interval from the local CSV's first and last line only; the file itself is never uploaded."""
    source = get(manifest, "source", default={})
    column = get(preprocessing, "column_mapping", "timestamp", default="timestamp_ms")
    unit = get(preprocessing, "column_mapping", "timestamp_unit", default="ms")
    candidates = [ROOT/name for name in DATASET_NAMES]+([Path(source["path"])] if source.get("path") else [])
    path = next((p for p in candidates if p.is_file()
                 and p.stat().st_size == source.get("size_bytes", p.stat().st_size)), None)
    if path is None or unit not in ("s", "ms", "us", "ns"):
        return None
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8", "ignore").strip().split(",")
        first = handle.readline().decode("utf-8", "ignore").strip().split(",")
        handle.seek(max(0, path.stat().st_size-8192))
        last = handle.read().decode("utf-8", "ignore").strip().splitlines()[-1].split(",")
    if column not in header or len(last) != len(header):
        return None
    index, scale = header.index(column), {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[unit]
    return [datetime.fromtimestamp(int(float(row[index]))/scale, tz=timezone.utc).isoformat()
            for row in (first, last)]


def repair_note(repairs):
    """row_repairs once a checkpoint records it; before that, the suite's recorded prose."""
    if isinstance(repairs, str):
        return repairs
    return (f"sorted by timestamp: {yes_no(repairs.get('sorted_by_timestamp'))} "
            f"({cell(repairs.get('rows_moved_by_sort'), ',')} rows moved); duplicate timestamps: "
            f"{cell(repairs.get('duplicate_timestamp_policy'))}, "
            f"{cell(repairs.get('duplicate_rows_dropped'), ',')} rows dropped; "
            f"{cell(repairs.get('rows_in_file'), ',')} -> {cell(repairs.get('rows_used'), ',')} rows")


def parameter_count(run, frozen=None):
    for source in (run["summary"], run["training"], run["config"]):
        for key in ("parameters", "parameter_count"):
            if isinstance(get(source, key), int):
                return get(source, key)
    if run["model"] == "e0":
        return 0
    # A shipped config.json is the architecture itself, so measure it rather than quote the
    # contract; the frozen count only stands in while this suite has no checkpoint at all.
    if not run["config"]:
        return get(frozen, "PARAMETERS", run["model"])
    spec = dict(run["config"])
    model = build_model(spec.pop("name"), spec.pop("history_rows"), spec.pop("channels"), **spec)
    total = sum(p.numel() for p in model.parameters())
    del model
    return total


def vast_reports(suite):
    folder = ROOT/suite["reports"]
    return {p.name: load(p) for p in sorted(folder.glob("*.json"))} if folder.is_dir() else {}


def find_key(obj, *names):
    """Schema of reports/vast/*.json belongs to the training driver; search it instead of assuming."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in names and isinstance(value, (str, int, float)):
                return value
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    else:
        return None
    return next((found for found in (find_key(child, *names) for child in children) if found is not None), None)


def summarize(value, folder, limit=3000):
    if value is None:
        return None
    for key in ("selected", "decision", "chosen", "schedule", "groups"):
        if isinstance(value, dict) and key in value:
            value = {key: value[key]} | {k: v for k, v in value.items()
                                         if k in ("reason", "decided_by", "policy", "frozen",
                                                  "aggregate_samples_per_second", "gpu_utilization_percent")}
            break
    text = json.dumps(value, indent=2)
    return text if len(text) <= limit else text[:limit]+f"\n... (truncated; full file at {folder}/)"


def table(header, rows):
    return ["| "+" | ".join(header)+" |", "|"+"|".join(["---"]*len(header))+"|",
            *["| "+" | ".join(str(c) for c in row)+" |" for row in rows]]


def metric_table(runs, split, unit="", spec=".6e", nested=None):
    rows = []
    for run in runs:
        metrics = run["metrics"].get(split)
        metrics = get(metrics, nested) if nested else metrics
        for k, horizon in enumerate(HORIZON_LABELS):
            rows.append((f"`{run['model']}`", horizon,
                         cell(get(metrics, "rmse", default=[None]*3)[k], spec),
                         cell(get(metrics, "mae", default=[None]*3)[k], spec),
                         cell(get(metrics, "r2", default=[None]*3)[k], ".6f"),
                         cell(get(metrics, "rmse_gain_vs_e0", default=[None]*3)[k], "+.6f"),
                         cell(get(metrics, "samples"))))
    return table(("Model", "Horizon", f"RMSE{unit}", f"MAE{unit}", "R2", "RMSE gain vs E0", "Samples"),
                 rows)


def metric_section(runs, split, suite):
    """Headline table in whatever unit the metric files declare, then their nested log-return copy."""
    units = {get(run["metrics"].get(split), "units") for run in runs if run["metrics"].get(split)}
    if len(units) > 1:
        raise RuntimeError(f"{split}: metric files disagree about units {sorted(map(str, units))}")
    # A metric file states its own space; the suite only predicts one no file has declared yet.
    price = units.pop() == "quote_currency" if units else suite["price_metrics"]
    lines = metric_table(runs, split, " (USD)" if price else "", ",.4f" if price else ".6e")
    if any(get(run["metrics"].get(split), "log_return") for run in runs):
        lines += ["", "The same predictions scored in log-return space (the `log_return` block of each "
                  "`*_metrics.json`), for comparison with the 60s run:", "",
                  *metric_table(runs, split, nested="log_return")]
    return lines


def suite_sections(suite, frozen):
    """Prose no metadata file carries: the metric space and the dataset's recorded repair."""
    parameters = ", ".join(f"`{model}` {count:,}"  # e0 is untrained, so it carries no count
                           for model, count in get(frozen, "PARAMETERS", default={}).items() if count)
    return [line for heading, body in suite["sections"]
            for line in (f"## {heading}", "",
                         body.format(parameters=parameters or "see the per-model tables below"), "")]


def model_card(repo_id, runs, plan, suite):
    shared = contract(runs)
    # The frozen contract JSON states the gate values before this suite's first checkpoint exists,
    # so the card is complete even when it is generated ahead of the run.
    frozen = (load(ROOT/suite["contract"]) if suite["contract"] else None) or {}
    fact = lambda value, key: value if value is not None else frozen.get(key)
    data = get(shared["experiment"], "data", default={})
    training = get(shared["experiment"], "training", default={})
    prep, target = shared["preprocessing"], shared["target_config"]
    split, vast = shared["split_manifest"], vast_reports(suite)
    stats = shared["data_stats"] or frozen.get("DATA_STATS")
    # Sorted rows mean the file's first and last line say nothing about the interval.
    interval = suite.get("interval") if data.get("sort_by_timestamp") else dataset_interval(split, prep)
    ranges = get(split, "ranges") or frozen.get("RANGES", {})
    counts = get(split, "sample_counts") or frozen.get("SAMPLE_COUNTS", {})
    repairs = get(split, "source", "row_repairs") or suite.get("row_repairs")
    gpus = (sorted({get(run, "training", "gpu") for run in runs if get(run, "training", "gpu")})
            or [find_key(vast, "gpu_name", "gpu", "device")])
    precisions = (sorted({get(run, "training", "precision") for run in runs
                          if get(run, "training", "precision")})
                  or [find_key(vast, "selected_precision") or training.get("precision")])
    source_path = get(split, "source", "path")
    host = (find_key(vast, "hostname", "host")
            or next((get(run, "training", "host") for run in runs if get(run, "training", "host")), None))
    versions = ", ".join(f"{k} {v}" for k, v in (shared["environment"] or {}).items())
    gaps = (f"gaps > 2s: {cell(get(stats, 'gaps_gt_2_seconds'))}"
            + (f", gaps > max_gap: {cell(get(stats, 'gaps_gt_max_gap'))}"
               if get(stats, "gaps_gt_max_gap") is not None else ""))
    lines = ["---", "license: mit", "library_name: pytorch", "tags:", *[f"- {t}" for t in TAGS], "---", "",
             f"# {suite['title']}", "", suite["intro"], "", "## Experiment contract", ""]
    lines += table(("Field", "Value"), [row for row in [
        ("Git commit", f"`{cell(git_commit())}`"),
        ("Dataset", f"`{Path(source_path).name if source_path else suite['csv']}` "
                    "- L10 depth: 10 bid + 10 ask levels, price and quantity per level"),
        ("Dataset SHA256", f"`{cell(fact(get(split, 'source', 'sha256'), 'CSV_SHA256'))}`"),
        ("Dataset rows", cell(get(stats, "rows"), ",")),
        ("Dataset interval (UTC)", " -> ".join(interval) if interval else "n/a"),
        # Only a suite that repairs its row order has anything to declare here.
        ("Row repair", repair_note(repairs)) if repairs else None,
        ("Sampling", f"median dt {cell(get(stats, 'median_dt_seconds'), suffix='s')}, "
                     f"p99 {cell(get(stats, 'p99_dt_seconds'), suffix='s')}, "
                     f"max {cell(get(stats, 'max_dt_seconds'), suffix='s')}, {gaps}, "
                     f"segment transitions: {cell(get(stats, 'segment_transitions'))}"),
        ("History", f"{cell(fact(get(prep, 'history_seconds'), 'HISTORY_SECONDS'), suffix='s')} -> "
                    f"{cell(fact(get(prep, 'history_rows'), 'HISTORY_ROWS'))} rows"
                    + (f" ({cell(get(prep, 'history_rounding'))}, median train continuous dt "
                       f"{cell(get(prep, 'median_train_dt_seconds'), suffix='s')})" if prep else "")),
        ("Stride", f"{cell(fact(get(prep, 'stride_seconds'), 'STRIDE_SECONDS'), suffix='s')} -> "
                   f"{cell(fact(get(prep, 'stride_rows'), 'STRIDE_ROWS'))} rows"),
        ("Horizons", " / ".join(f"{h}s" for h in (get(target, "horizons_seconds")
                                                  or frozen.get("HORIZONS", []))) or "n/a"),
        ("Target", f"{cell(get(target, 'definition'))}; {cell(get(target, 'lookup'))}; "
                   f"tolerance {cell(fact(get(target, 'tolerance_seconds'), 'TOLERANCE_SECONDS'), suffix='s')}"),
        ("Split boundaries (exclusive, UTC)",
         " ".join(f"{k}={v}" for k, v in zip(("train_end", "validation_end"),
                                             get(split, "boundaries_utc", default=["n/a", "n/a"])))),
        ("Split row ranges", " ".join(f"{k}=[{v[0]}, {v[1]})" for k, v in ranges.items()) or "n/a"),
        ("Split samples", " ".join(f"{k}={v}" for k, v in counts.items()) or "n/a"),
        ("Split policy", cell(get(split, "policy"))),
        ("Gap rule", f"max_gap_seconds={cell(fact(get(target, 'max_gap_seconds'), 'MAX_GAP_SECONDS'))}"
                     ", segment boundary rejection="
                     f"{cell(get(target, 'segment_boundary_rejection'))}; a window is dropped if any gap or "
                     "segment change falls between its first history row and its last target row"),
        ("Batch size", cell(training.get("batch_size"))),
        ("Epochs", cell(training.get("epochs"))),
        ("Optimizer", f"{OPTIMIZER}; lr={cell(training.get('learning_rate'))}, "
                      f"weight_decay={cell(training.get('weight_decay'))}, "
                      f"gradient clip {cell(training.get('gradient_clip'))}, "
                      f"gradient accumulation {cell(training.get('gradient_accumulation'))}"),
        ("Loss", f"{cell(training.get('loss'))} on unscaled log returns; checkpoint selection = "
                 "validation mean MSE across the 3 horizons (test never used)"),
        ("Seed", cell(training.get("seed"))),
        ("Precision", f"{', '.join(p for p in precisions if p) or 'n/a'} autocast, TF32 matmul/cuDNN enabled"),
        ("torch.compile", " ".join(f"{run['model']}={yes_no(get(run, 'training', 'compile'))}"
                                   for run in runs if run["model"] != "e0") or "n/a"),
        ("Hardware", f"{', '.join(g for g in gpus if g) or 'n/a'} on host {cell(host)}; CUDA runtime "
                     f"{cell(find_key(vast, 'cuda_runtime_version', 'cuda_version'))}, "
                     f"driver {cell(find_key(vast, 'driver_version', 'driver'))}, "
                     f"VRAM {cell(find_key(vast, 'vram_total_bytes'), ',')} bytes"),
        ("Library versions", versions or "n/a"),
    ] if row])
    lines += ["", *suite_sections(suite, frozen), "## Models", ""]
    for run in runs:
        prep_run, schema = run["preprocessing"], run["feature_schema"]
        shape = get(schema, "sample_shape", default=[])
        lines += [f"### `{run['model']}` - `{run['run']}`", ""]
        if run["model"] == "e0":
            lines += [f"{suite['e0_note']} Parameters: {cell(parameter_count(run, frozen))}. "
                      "No checkpoint; predictions and metrics only.", ""]
            continue
        lines += table(("Field", "Value"), [
            ("Parameters", cell(parameter_count(run, frozen), ",")),
            ("Input", f"[{cell(get(run, 'config', 'history_rows', default=get(prep_run, 'history_rows')))}"
                      f", {', '.join(str(d) for d in shape) or 'n/a'}] "
                      f"({cell(get(run, 'config', 'channels'))} channels)"),
            ("Features", cell(len(get(schema, "names", default=[])) or None)),
            ("Normalization", cell(get(schema, "normalization"))),
            ("Best epoch", f"{cell(get(run, 'training', 'best_epoch'))} (validation mean MSE "
                           f"{cell(get(run, 'training', 'best_validation_mean_mse'), '.6e')})"),
            ("Epochs completed", cell(get(run, "training", "epochs_completed"))),
            ("Batch size / workers", f"{cell(get(run, 'training', 'batch_size'))} / "
                                     f"{cell(get(run, 'training', 'num_workers'))}"),
            ("Precision / device / torch.compile",
             f"{cell(get(run, 'training', 'precision'))} / {cell(get(run, 'training', 'device'))} / "
             f"{yes_no(get(run, 'training', 'compile'))}"),
            ("Training wall time", f"{cell(get(run, 'training', 'total_seconds'), ',.1f')} s"),
            ("Peak VRAM (allocated / reserved)",
             f"{cell(get(run, 'training', 'peak_allocated_bytes'), ',')} / "
             f"{cell(get(run, 'training', 'peak_reserved_bytes'), ',')} bytes"),
        ])
        if run["config"]:
            lines += ["", "Architecture (`config.json`):", "", "```json",
                      json.dumps(run["config"], indent=2), "```"]
        lines += [""]
    lines += ["## Validation metrics", "", *metric_section(runs, "validation", suite), "",
              "## Test metrics", "",
              "Test was scored once, after training and best-checkpoint selection completed.", "",
              *metric_section(runs, "test", suite), "",
              "## Train metrics", "", *metric_section(runs, "train", suite), "",
              "## Concurrency schedule and throughput", ""]
    for name in ("concurrency_benchmark.json", "training_schedule.json"):
        body = summarize(vast.get(name), suite["reports"])
        lines += [f"`{suite['reports']}/{name}`:", "", "```json", body, "```", ""] if body else \
                 [f"`{suite['reports']}/{name}`: not present.", ""]
    benchmark = get(vast.get("single_job_benchmark.json"), "models", default={})
    if benchmark:
        lines += [f"Single-job benchmark (`{suite['reports']}/single_job_benchmark.json`), batch size "
                  f"{cell(get(vast['single_job_benchmark.json'], 'batch_size'))}:", "",
                  *table(("Model", "Samples/s", "Mean step (s)", "Peak allocated (bytes)", "Avg GPU util %"),
                         [(f"`{name}`", cell(get(value, "samples_per_second"), ",.1f"),
                           cell(get(value, "mean_step_seconds"), ".6f"),
                           cell(get(value, "peak_allocated_bytes"), ","),
                           cell(get(value, "average_gpu_utilization_percent"), ".1f"))
                          for name, value in benchmark.items()]), ""]
    if (ROOT/suite["reports"]/"gpu_usage.csv").is_file():
        shipped = any(name.endswith(".csv") for item in plan for name in item["files"])
        lines += [f"GPU sampling log `{suite['reports']}/gpu_usage.csv` is "
                  f"{'included' if shipped else 'kept local'}; the raw book CSV is never uploaded.", ""]
    example = "/".join(RUNS[-1])  # any run would do; lit is the last one listed above
    lines += ["## Repository layout", "", "```text"]
    lines += [f"{item['repo_dir']}/  ({len(item['files'])} files, {human(item['bytes'])})" for item in plan]
    lines += ["```", "",
              "Prediction tables carry one row per origin, ordered by origin timestamp, with "
              "`origin_mid`, per-horizon target index/timestamp/mid, `true_return_h`, `pred_return_h` and "
              "`pred_mid_h = origin_mid * exp(pred_return_h)`, so every plot can be rebuilt "
              "without the model.", "",
              "## Loading a checkpoint", "", "```python",
              "from huggingface_hub import snapshot_download",
              "from src.training.checkpoint import from_pretrained  # this repo's code, not AutoModel", "",
              f'path = snapshot_download("{repo_id}", allow_patterns="{example}/best/*")',
              f'model = from_pretrained(f"{{path}}/{example}/best")',
              "```", "",
              "`best/` also carries `optimizer.pt`, `scheduler.pt` and `trainer_state.pt`; `last/` resumes "
              "the exact training run at an epoch boundary. No adapter is merged into these base weights. "
              "HFformer and LiT expose `q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2` as separate "
              "Linear modules for later LoRA injection; PatchTST is the stock "
              "`transformers.PatchTSTModel`, so its four unmerged attention projections carry those "
              "names while its feed-forward Linears stay at the upstream `ff.0`/`ff.3` paths.", ""]
    return "\n".join(lines)


def write_card(path, card):
    # --card-out is a preview sink; the repo README is source, not a generated artifact.
    if Path(path).resolve() == ROOT/"README.md":
        raise SystemExit(f"--card-out must not overwrite {ROOT/'README.md'}")
    Path(path).write_text(card)


def print_plan(repo_id, plan, missing):
    print(f"destination: {repo_id} (model repo, private)")
    total = 0
    for item in plan:
        print(f"\n{item['repo_dir']}/  <- {item['local']}  [{item['message']}]")
        for name in item["files"]:
            print(f"  {human((item['local']/name).stat().st_size):>10}  {item['repo_dir']}/{name}")
        for name in item["skipped"]:
            print(f"  {'excluded':>10}  {item['local']/name}")
        if not item["files"]:
            print("  (nothing to upload)")
        total += item["bytes"]
    print("\nREADME.md  <- generated model card")
    print(f"{sum(len(i['files']) for i in plan)} files, {human(total)}, "
          f"{sum(1 for i in plan if i['files'])+1} commits")
    for path in missing:
        print(f"MISSING REQUIRED: {path}")


def upload(api, repo_id, plan, card, allow_public=False):
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    # private=True only applies to a repo this call creates; an existing public one stays public.
    if not allow_public and getattr(api.repo_info(repo_id, repo_type="model"), "private", True) is False:
        raise SystemExit(f"{repo_id} already exists and is public; make it private or pass --allow-public")
    with tempfile.TemporaryDirectory(prefix="hf-card-") as tmp:
        path = Path(tmp)/"README.md"
        path.write_text(card)
        api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md", repo_id=repo_id,
                        repo_type="model", commit_message="Model card")
    for item in plan:
        if not item["files"]:
            print(f"skip {item['repo_dir']}/ (no files)")
            continue
        print(f"upload {item['repo_dir']}/ ({len(item['files'])} files, {human(item['bytes'])})", flush=True)
        # One commit per folder keeps a failed publish resumable without re-sending the rest.
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(item["local"]),
                          path_in_repo=item["repo_dir"], commit_message=item["message"],
                          allow_patterns=item["files"], ignore_patterns=item["ignore"])


def verify(api, repo_id, plan):
    remote = set(api.list_repo_files(repo_id, repo_type="model"))
    expected = required_paths() | {"README.md"} | {f"{i['repo_dir']}/{n}" for i in plan for n in i["files"]}
    groups = {}
    for path in sorted(expected):
        present, absent = groups.setdefault(str(Path(path).parent), ([], []))
        (present if path in remote else absent).append(path)
    print(f"\n{'STATUS':6} {'FILES':>9}  PATH")
    failures = []
    for group, (present, absent) in sorted(groups.items()):
        print(f"{'FAIL' if absent else 'PASS':6} {f'{len(present)}/{len(present)+len(absent)}':>9}  "
              f"{'(repo root)' if group == '.' else group+'/'}")
        failures += absent
    for path in failures:
        print(f"  missing: {path}")
    print(f"{len(expected)-len(failures)}/{len(expected)} expected files present in {repo_id}")
    return not failures


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=sorted(SUITES), default="oct2023",
                   help="Frozen experiment to publish: run names, destination repo and model card")
    p.add_argument("--repo-id", help="Overrides HF_REPO_ID; default is the suite's repo")
    p.add_argument("--card-out", help="Also write the generated model card here (never the repo README)")
    p.add_argument("--include-monitoring-csv", action="store_true",
                   help="Ship the suite's reports/*.csv GPU logs; the book CSV stays excluded either way")
    p.add_argument("--allow-public", action="store_true",
                   help="Permit publishing into a repo that already exists and is public")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the plan and card; no network calls")
    mode.add_argument("--verify-only", action="store_true", help="Only check the remote file list")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    suite = select_suite(args.suite)
    plan = build_plan(args.include_monitoring_csv)
    missing = missing_required(plan)
    runs = [gather(*E0_RUN)]+[gather(model, run) for model, run in RUNS]
    if args.dry_run:
        repo_id = (args.repo_id or os.environ.get("HF_REPO_ID")
                   or suite["repo"].format(user="<authenticated user>"))
        card = model_card(repo_id, runs, plan, suite)
        print_plan(repo_id, plan, missing)
        print("\n"+"="*80+"\nREADME.md\n"+"="*80)
        print(card)
        if args.card_out:
            write_card(args.card_out, card)
        return 0
    if missing and not args.verify_only:
        raise SystemExit("Missing required artifacts; train/export before publishing:\n  "+"\n  ".join(missing))
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("No Hugging Face token: set HF_TOKEN or log in with huggingface-cli")
    api = HfApi(token=token)
    repo_id = (args.repo_id or os.environ.get("HF_REPO_ID")
               or suite["repo"].format(user=api.whoami()["name"]))
    if not args.verify_only:
        card = model_card(repo_id, runs, plan, suite)
        if args.card_out:
            write_card(args.card_out, card)
        print_plan(repo_id, plan, missing)
        upload(api, repo_id, plan, card, args.allow_public)
    return 0 if verify(api, repo_id, plan) else 1


if __name__ == "__main__":
    sys.exit(main())
