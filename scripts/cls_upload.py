"""Publish classification artifacts to Hugging Face and read them back.

  python scripts/cls_upload.py upload   [--only-runs] [--no-runs]
  python scripts/cls_upload.py verify

Layout on Tson29/LOB_Classification_WF3 (mirrors runs/cls):

  <equal_width|quantile>/<multi|single>/<arch>/[h<seconds>/]f<fold>/   one run each
  followups/<experiment>/<variant>/<multi|single>/<arch>/[h<s>/]f<fold>/
  labeling/  reports/  audits/  benchmarks/  registry/  README.md

Per run: best/ (weights + config + every metadata JSON), run_config, label_set,
training_history, train/validation/test metrics, validation/test predictions (CSV) and
class probabilities (npz), run_summary, environment, git, status, the job log.
Never uploaded: the raw CSV (it stays in Tson29/btc-l10-gate-1y) and last/ folders.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(Path.home()/".hf_home"))
from huggingface_hub import HfApi, hf_hub_download

REPO = "Tson29/LOB_Classification_WF3"
RUN_FILES = ("run_config.json", "label_set.json", "training_history.jsonl", "train_metrics.json",
             "validation_metrics.json", "test_metrics.json", "validation_predictions.csv.gz",
             "test_predictions.csv.gz", "validation_probs.npz", "test_probs.npz", "run_summary.json",
             "environment.json", "git.json", "status.json")
TOP = dict(labeling="labeling", reports="reports", audits="audits", benchmarks="benchmarks")


def completed_runs():
    for summary in sorted((ROOT/"runs/cls").rglob("run_summary.json")):
        run = summary.parent
        if "benchmark_work" in run.parts or not (run/"best/model.safetensors").exists():
            continue
        yield run


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage(target, runs=True, meta=True):
    if meta:
        for name, local in TOP.items():
            for f in (ROOT/local).rglob("*"):
                if f.is_file() and "superseded" not in f.parts:
                    link(f, target/name/f.relative_to(ROOT/local))
        link(ROOT/"runs/cls/registry.json", target/"registry/registry.json")
        link(ROOT/"prompt.md", target/"reports/prompt.md")
        if (ROOT/"reports/HF_README.md").exists():
            link(ROOT/"reports/HF_README.md", target/"README.md")
    count = 0
    if runs:
        for run in completed_runs():
            rel = run.relative_to(ROOT/"runs/cls")
            for name in RUN_FILES:
                if (run/name).exists():
                    link(run/name, target/rel/name)
            for f in (run/"best").iterdir():
                link(f, target/rel/"best"/f.name)
            cfg = json.loads((run/"run_config.json").read_text())
            log = ROOT/"runs/cls/logs"/f"{cfg['job_id']}.log"
            if log.exists():
                link(log, target/rel/"train.log")
            count += 1
    return count


def upload(args):
    api = HfApi()
    with tempfile.TemporaryDirectory(dir=ROOT/"runs") as tmp:
        target = Path(tmp)/"hf"
        n = stage(target, runs=not args.no_runs, meta=not args.only_runs)
        print(f"staged {n} runs", flush=True)
        api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=str(target),
                                print_report=False)
    print("upload complete", flush=True)


def verify(args):
    """Download representative files and prove they are usable, not just present."""
    import numpy as np
    import pandas as pd
    import torch
    from src.cls.labels import LabelSet
    from src.cls.metrics import official_metrics
    from src.cls.models import load_classifier
    api = HfApi()
    files = set(api.list_repo_files(REPO))
    checks = []

    def check(name, ok, detail=None):
        checks.append(dict(name=name, passed=bool(ok), detail=detail or {}))
        print(("PASS " if ok else "FAIL ")+name, flush=True)

    runs = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("run_summary.json")})
    check("runs_present", len(runs) > 0, dict(runs=len(runs)))
    for required in ("reports/FINAL_REPORT_CLASSIFICATION.md", "audits/leakage_audit_classification.json",
                     "labeling/equal_width_k32_p99.json", "labeling/quantile_c34.json",
                     "benchmarks/concurrency_benchmark.json", "benchmarks/training_schedule.json",
                     "registry/registry.json", "README.md"):
        check(f"present:{required}", required in files)
    with tempfile.TemporaryDirectory() as tmp:
        get = lambda f: hf_hub_download(REPO, f, local_dir=tmp)
        if "reports/FINAL_REPORT_CLASSIFICATION.md" in files:
            text = Path(get("reports/FINAL_REPORT_CLASSIFICATION.md")).read_text()
            check("report_readable", "Final conclusions" in text, dict(chars=len(text)))
        if "audits/leakage_audit_classification.json" in files:
            audit = json.loads(Path(get("audits/leakage_audit_classification.json")).read_text())
            check("audit_readable_and_pass", audit["status"] == "pass",
                  dict(passed=audit["checks_passed"], total=audit["checks_total"]))
        for name in ("equal_width_k32_p99", "quantile_c34"):
            if f"labeling/{name}.json" in files:
                ls = LabelSet.load(get(f"labeling/{name}.json"))  # verifies its own fingerprint
                check(f"labels_readable:{name}", ls[60].n_classes > 0, dict(sha256=ls.sha256()))
        sample = [r for r in runs if "/multi/" in r][:3]+[r for r in runs if "/single/" in r][:2]
        for run in sample:
            summary = json.loads(Path(get(f"{run}/run_summary.json")).read_text())
            metrics = json.loads(Path(get(f"{run}/test_metrics.json")).read_text())["official_metrics"]
            frame = pd.read_csv(get(f"{run}/test_predictions.csv.gz"))
            ok = True
            for tag, m in metrics.items():
                r, _ = official_metrics(frame["origin_mid"], frame[f"target_mid_{tag}"], frame[f"pred_mid_{tag}"])
                ok &= all(abs(r[k]-m[k]) <= 1e-9*max(1, abs(m[k])) for k in r)
            check(f"predictions_recompute_metrics:{run}", ok and len(frame) == 235848, dict(rows=len(frame)))
            for f in ("model.safetensors", "config.json"):
                get(f"{run}/best/{f}")
            model = load_classifier(Path(tmp)/run/"best")
            x = torch.randn(4, summary["history_rows"], model.model_config["channels"])
            out = model(x)
            check(f"checkpoint_loads_and_runs:{run}", all(torch.isfinite(o).all() for o in out),
                  dict(parameters=sum(p.numel() for p in model.parameters())))
            probs = np.load(get(f"{run}/test_probs.npz"))
            check(f"probabilities_readable:{run}", all(np.isclose(probs[k].astype(np.float64).sum(1), 1, atol=5e-3).all()
                                                       for k in probs.files if k.endswith("s")))
    failed = [c["name"] for c in checks if not c["passed"]]
    report = dict(repo=REPO, status="fail" if failed else "pass", files=len(files), runs=len(runs),
                  failed=failed, checks=checks)
    out = ROOT/"audits/hf_readback_verification.json"
    out.write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(checks)-len(failed)}/{len(checks)} read-back checks pass; {out}")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("upload", "verify"))
    p.add_argument("--only-runs", action="store_true")
    p.add_argument("--no-runs", action="store_true")
    args = p.parse_args()
    return upload(args) if args.mode == "upload" else verify(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
