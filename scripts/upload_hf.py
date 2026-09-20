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
E0_RUN = ("e0", "e0_60s")
RUNS = (("ofi_lstm", "ofi_lstm_60s_base"), ("hfformer", "hfformer_60s_base"),
        ("patchtst", "patchtst_60s_base"), ("moderntcn", "moderntcn_60s_base"), ("lit", "lit_60s_base"))
SPLITS = ("train", "validation", "test")
CHECKPOINT_FILES = (("model.safetensors", "config.json", "environment.json",
                     "optimizer.pt", "scheduler.pt", "trainer_state.pt")
                    + tuple(f"{name}.json" for name in METADATA_FILES))
ARTIFACT_FILES = (tuple(f"{s}_predictions.csv.gz" for s in SPLITS)
                  + tuple(f"{s}_metrics.json" for s in SPLITS) + ("run_summary.json",))
# The raw L10 book CSV never leaves the machine; only gzipped prediction tables ship.
IGNORE = ("*.csv", "*.pyc", "*__pycache__/*", "*.venv/*", "*.previous/*", ".*", "*/.*")
DATASET_NAMES = ("BTCUSDT_L10_oct2023.csv",)
TAGS = ("time-series-forecasting", "limit-order-book", "market-microstructure", "bitcoin", "btcusdt")
OPTIMIZER = "AdamW(lr, weight_decay) + CosineAnnealingLR(T_max=epochs), no warmup, no scheduler tuning"


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
    # reports/vast/gpu_usage.csv is the only CSV that may ship, and only on request.
    reports_ignore = (tuple(p for p in IGNORE if p != "*.csv")+("*BTCUSDT*.csv",)) if monitoring_csv else IGNORE
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


def parameter_count(run):
    for source in (run["summary"], run["training"], run["config"]):
        for key in ("parameters", "parameter_count"):
            if isinstance(get(source, key), int):
                return get(source, key)
    if run["model"] == "e0":
        return 0
    if not run["config"]:
        return None
    spec = dict(run["config"])
    model = build_model(spec.pop("name"), spec.pop("history_rows"), spec.pop("channels"), **spec)
    total = sum(p.numel() for p in model.parameters())
    del model
    return total


def vast_reports():
    folder = ROOT/"reports"/"vast"
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


def summarize(value, limit=3000):
    if value is None:
        return None
    for key in ("selected", "decision", "chosen", "schedule", "groups"):
        if isinstance(value, dict) and key in value:
            value = {key: value[key]} | {k: v for k, v in value.items()
                                         if k in ("reason", "decided_by", "policy", "frozen",
                                                  "aggregate_samples_per_second", "gpu_utilization_percent")}
            break
    text = json.dumps(value, indent=2)
    return text if len(text) <= limit else text[:limit]+"\n... (truncated; full file at reports/vast/)"


def table(header, rows):
    return ["| "+" | ".join(header)+" |", "|"+"|".join(["---"]*len(header))+"|",
            *["| "+" | ".join(str(c) for c in row)+" |" for row in rows]]


def metric_table(runs, split):
    rows = []
    for run in runs:
        metrics = run["metrics"].get(split)
        for k, horizon in enumerate(HORIZON_LABELS):
            rows.append((f"`{run['model']}`", horizon,
                         cell(get(metrics, "rmse", default=[None]*3)[k], ".6e"),
                         cell(get(metrics, "mae", default=[None]*3)[k], ".6e"),
                         cell(get(metrics, "r2", default=[None]*3)[k], ".6f"),
                         cell(get(metrics, "rmse_gain_vs_e0", default=[None]*3)[k], "+.6f"),
                         cell(get(metrics, "samples"))))
    return table(("Model", "Horizon", "RMSE", "MAE", "R2", "RMSE gain vs E0", "Samples"), rows)


def model_card(repo_id, runs, plan):
    shared = contract(runs)
    data = get(shared["experiment"], "data", default={})
    training = get(shared["experiment"], "training", default={})
    prep, target = shared["preprocessing"], shared["target_config"]
    split, stats, vast = shared["split_manifest"], shared["data_stats"], vast_reports()
    interval = dataset_interval(split, prep)
    ranges, counts = get(split, "ranges", default={}), get(split, "sample_counts", default={})
    gpus = (sorted({get(run, "training", "gpu") for run in runs if get(run, "training", "gpu")})
            or [find_key(vast, "gpu_name", "gpu", "device")])
    precisions = (sorted({get(run, "training", "precision") for run in runs
                          if get(run, "training", "precision")})
                  or [find_key(vast, "selected_precision") or training.get("precision")])
    source_path = get(split, "source", "path")
    host = (find_key(vast, "hostname", "host")
            or next((get(run, "training", "host") for run in runs if get(run, "training", "host")), None))
    versions = ", ".join(f"{k} {v}" for k, v in (shared["environment"] or {}).items())
    lines = ["---", "license: mit", "library_name: pytorch", "tags:", *[f"- {t}" for t in TAGS], "---", "",
             "# BTCUSDT L10 multi-horizon log-return forecasting (60s history)",
             "",
             "Five learned models plus the untrained `e0` zero-return baseline, trained once under a "
             "frozen experiment contract on a single October 2023 BTCUSDT L10 limit-order-book file. "
             "Every checkpoint (`best`, `last`), every prediction table and every metric file below "
             "was produced by that single run; nothing here is tuned, re-fitted or re-scored.", "",
             "## Experiment contract", ""]
    lines += table(("Field", "Value"), [
        ("Git commit", f"`{cell(git_commit())}`"),
        ("Dataset", f"`{Path(source_path).name if source_path else 'n/a'}` "
                    "- L10 depth: 10 bid + 10 ask levels, price and quantity per level"),
        ("Dataset SHA256", f"`{cell(get(split, 'source', 'sha256'))}`"),
        ("Dataset rows", cell(get(stats, "rows"), ",")),
        ("Dataset interval (UTC)", " -> ".join(interval) if interval else "n/a"),
        ("Sampling", f"median dt {cell(get(stats, 'median_dt_seconds'), suffix='s')}, "
                     f"p99 {cell(get(stats, 'p99_dt_seconds'), suffix='s')}, "
                     f"max {cell(get(stats, 'max_dt_seconds'), suffix='s')}, "
                     f"gaps > 2s: {cell(get(stats, 'gaps_gt_2_seconds'))}, "
                     f"segment transitions: {cell(get(stats, 'segment_transitions'))}"),
        ("History", f"{cell(get(prep, 'history_seconds'), suffix='s')} -> "
                    f"{cell(get(prep, 'history_rows'))} rows "
                    f"({cell(get(prep, 'history_rounding'))}, median train continuous dt "
                    f"{cell(get(prep, 'median_train_dt_seconds'), suffix='s')})"),
        ("Stride", f"{cell(get(prep, 'stride_seconds'), suffix='s')} -> {cell(get(prep, 'stride_rows'))} rows"),
        ("Horizons", " / ".join(f"{h}s" for h in get(target, "horizons_seconds", default=[])) or "n/a"),
        ("Target", f"{cell(get(target, 'definition'))}; {cell(get(target, 'lookup'))}; "
                   f"tolerance {cell(get(target, 'tolerance_seconds'), suffix='s')}"),
        ("Split boundaries (exclusive, UTC)",
         " ".join(f"{k}={v}" for k, v in zip(("train_end", "validation_end"),
                                             get(split, "boundaries_utc", default=["n/a", "n/a"])))),
        ("Split row ranges", " ".join(f"{k}=[{v[0]}, {v[1]})" for k, v in ranges.items()) or "n/a"),
        ("Split samples", " ".join(f"{k}={v}" for k, v in counts.items()) or "n/a"),
        ("Split policy", cell(get(split, "policy"))),
        ("Gap rule", f"max_gap_seconds={cell(get(target, 'max_gap_seconds'))}, segment boundary rejection="
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
    ])
    lines += ["", "## Models", ""]
    for run in runs:
        prep_run, schema = run["preprocessing"], run["feature_schema"]
        shape = get(schema, "sample_shape", default=[])
        lines += [f"### `{run['model']}` - `{run['run']}`", ""]
        if run["model"] == "e0":
            lines += ["Untrained reference baseline: predicts a zero log return at every horizon, so "
                      "RMSE_E0 = sqrt(mean(y^2)) and every gain below is measured against it. Parameters: "
                      f"{cell(parameter_count(run))}. No checkpoint; predictions and metrics only.", ""]
            continue
        lines += table(("Field", "Value"), [
            ("Parameters", cell(parameter_count(run), ",")),
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
    lines += ["## Validation metrics", "", *metric_table(runs, "validation"), "",
              "## Test metrics", "",
              "Test was scored once, after training and best-checkpoint selection completed.", "",
              *metric_table(runs, "test"), "",
              "## Train metrics", "", *metric_table(runs, "train"), "",
              "## Concurrency schedule and throughput", ""]
    for name in ("concurrency_benchmark.json", "training_schedule.json"):
        body = summarize(vast.get(name))
        lines += [f"`reports/vast/{name}`:", "", "```json", body, "```", ""] if body else \
                 [f"`reports/vast/{name}`: not present.", ""]
    benchmark = get(vast.get("single_job_benchmark.json"), "models", default={})
    if benchmark:
        lines += ["Single-job benchmark (`reports/vast/single_job_benchmark.json`), batch size "
                  f"{cell(get(vast['single_job_benchmark.json'], 'batch_size'))}:", "",
                  *table(("Model", "Samples/s", "Mean step (s)", "Peak allocated (bytes)", "Avg GPU util %"),
                         [(f"`{name}`", cell(get(value, "samples_per_second"), ",.1f"),
                           cell(get(value, "mean_step_seconds"), ".6f"),
                           cell(get(value, "peak_allocated_bytes"), ","),
                           cell(get(value, "average_gpu_utilization_percent"), ".1f"))
                          for name, value in benchmark.items()]), ""]
    if (ROOT/"reports"/"vast"/"gpu_usage.csv").is_file():
        shipped = any(name.endswith(".csv") for item in plan for name in item["files"])
        lines += [f"GPU sampling log `reports/vast/gpu_usage.csv` is {'included' if shipped else 'kept local'}; "
                  "the raw book CSV is never uploaded.", ""]
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
              f'path = snapshot_download("{repo_id}", allow_patterns="lit/lit_60s_base/best/*")',
              'model = from_pretrained(f"{path}/lit/lit_60s_base/best")',
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
    p.add_argument("--repo-id", help="Overrides HF_REPO_ID; default <authenticated user>/Pretrain_Model")
    p.add_argument("--card-out", help="Also write the generated model card here (never the repo README)")
    p.add_argument("--include-monitoring-csv", action="store_true",
                   help="Ship reports/vast/*.csv GPU logs; the book CSV stays excluded either way")
    p.add_argument("--allow-public", action="store_true",
                   help="Permit publishing into a repo that already exists and is public")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the plan and card; no network calls")
    mode.add_argument("--verify-only", action="store_true", help="Only check the remote file list")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args.include_monitoring_csv)
    missing = missing_required(plan)
    runs = [gather(*E0_RUN)]+[gather(model, run) for model, run in RUNS]
    if args.dry_run:
        repo_id = args.repo_id or os.environ.get("HF_REPO_ID") or "<authenticated user>/Pretrain_Model"
        card = model_card(repo_id, runs, plan)
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
    repo_id = args.repo_id or os.environ.get("HF_REPO_ID") or f"{api.whoami()['name']}/Pretrain_Model"
    if not args.verify_only:
        card = model_card(repo_id, runs, plan)
        if args.card_out:
            write_card(args.card_out, card)
        print_plan(repo_id, plan, missing)
        upload(api, repo_id, plan, card, args.allow_public)
    return 0 if verify(api, repo_id, plan) else 1


if __name__ == "__main__":
    sys.exit(main())
