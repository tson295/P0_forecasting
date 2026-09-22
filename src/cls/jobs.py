"""Job specs, run configs and the machine-readable run registry.

A job is one trained model: (method or label set, formulation, architecture, horizons,
fold, variant). Its run directory mirrors the Hugging Face layout:

  runs/cls/<method>/<multi|single>/<arch>/[h<seconds>/]f<fold>/
  runs/cls/followups/<experiment>/<variant>/<arch>/[h<seconds>/]f<fold>/

The registry (runs/cls/registry.json) is owned by the scheduler; every status change is
written atomically. Statuses: planned -> running -> completed | failed; a failed job
that is re-queued keeps its attempt history and is flagged retried.
"""
import copy
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import HORIZONS

ROOT = Path(__file__).resolve().parents[2]
ARCHS = ("ofi_lstm", "moderntcn", "transformer")
METHOD_LABELS = {"equal_width": "labeling/equal_width_k32_p99.json",
                 "quantile": "labeling/quantile_c34.json"}
METHOD_SHORT = {"equal_width": "ew", "quantile": "q"}

# The WF3 recipe (configs/*_wf3_f*.json): AdamW lr 1e-4 wd 1e-4, cosine T_max=epochs,
# 30 epochs, batch 128, clip 1.0, bf16, seed 42. Compilation follows WF3 where WF3 had
# the model (ofi_lstm eager, moderntcn compiled); the Transformer adds CUDA graphs.
TRAINING = dict(epochs=30, batch_size=128, learning_rate=1e-4, weight_decay=1e-4,
                gradient_clip=1.0, precision="bf16", seed=42, eval_batch_size=1024,
                optimizer="AdamW(fused=True)", scheduler="CosineAnnealingLR(T_max=epochs)",
                selection="validation mean CrossEntropy")
COMPILE = dict(ofi_lstm="none", moderntcn="default", transformer="reduce-overhead")


def wf3_data_fields(fold):
    return json.loads((ROOT/f"configs/moderntcn_wf3_f{fold}.json").read_text())["data"]


def job_id(method, formulation, arch, horizons, fold, experiment=None, variant=None):
    htag = "all" if len(horizons) == 3 else f"h{horizons[0]}"
    core = f"{METHOD_SHORT.get(method, method)}-{formulation}-{arch}-{htag}-f{fold}"
    return core if experiment is None else f"fu-{experiment}-{variant}-{core}"


def run_dir(method, formulation, arch, horizons, fold, experiment=None, variant=None):
    parts = ["runs", "cls"]
    parts += [method] if experiment is None else ["followups", experiment, variant]
    parts += [formulation, arch]
    if len(horizons) == 1:
        parts.append(f"h{horizons[0]}")
    parts.append(f"f{fold}")
    return str(Path(*parts))


def make_job(method, formulation, arch, horizons, fold, stage, experiment=None, variant=None,
             label_set_path=None, model_kwargs=None, training=None, extra=None):
    horizons = [int(h) for h in horizons]
    if formulation == "multi" and horizons != list(HORIZONS):
        raise ValueError("multi-horizon jobs predict 60/120/180 s")
    if formulation == "single" and len(horizons) != 1:
        raise ValueError("single-horizon jobs predict one horizon")
    return dict(id=job_id(method, formulation, arch, horizons, fold, experiment, variant),
                stage=stage, experiment=experiment, variant=variant or "baseline",
                method=method, formulation=formulation, arch=arch, horizons=horizons, fold=int(fold),
                label_set_path=label_set_path or METHOD_LABELS[method],
                model_kwargs=model_kwargs or {}, training=TRAINING | {"compile_mode": COMPILE[arch]} | (training or {}),
                extra=extra or {},
                run_dir=run_dir(method, formulation, arch, horizons, fold, experiment, variant),
                status="planned", attempts=[], retried=False)


def baseline_jobs():
    """The required 72: 2 methods x 3 archs x 3 folds x (multi + 3 single horizons)."""
    jobs = []
    for method in ("equal_width", "quantile"):
        for arch in ARCHS:
            for fold in (1, 2, 3):
                jobs.append(make_job(method, "multi", arch, HORIZONS, fold, stage="multi"))
    for method in ("equal_width", "quantile"):
        for arch in ARCHS:
            for h in HORIZONS:
                for fold in (1, 2, 3):
                    jobs.append(make_job(method, "single", arch, [h], fold, stage="single"))
    return jobs


def run_config(job, csv_path):
    """The complete, self-contained config a job's process reads (saved in its run dir)."""
    return dict(job_id=job["id"], stage=job["stage"], experiment=job.get("experiment"),
                variant=job.get("variant", "baseline"), method=job["method"],
                formulation=job["formulation"], arch=job["arch"], horizons=job["horizons"],
                fold=job["fold"], run_dir=job["run_dir"], label_set_path=str(ROOT/job["label_set_path"]),
                data=wf3_data_fields(job["fold"]), csv_path=csv_path,
                training=copy.deepcopy(job["training"]), model_kwargs=copy.deepcopy(job["model_kwargs"]),
                extra=copy.deepcopy(job.get("extra", {})))


def now():
    return pd.Timestamp.now(tz="UTC").isoformat()


class Registry:
    def __init__(self, path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(".lock")

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            yield data
            self.write(data)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            return dict(created=now(), jobs=[])
        return json.loads(self.path.read_text())

    def write(self, data):
        data["updated"] = now()
        counts = {}
        for job in data["jobs"]:
            counts[job["status"]] = counts.get(job["status"], 0)+1
        data["counts"] = counts
        data["retried"] = [j["id"] for j in data["jobs"] if j.get("retried")]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2)+"\n")
        tmp.replace(self.path)

    def add(self, jobs):
        with self.locked() as data:
            known = {j["id"] for j in data["jobs"]}
            added = [j for j in jobs if j["id"] not in known]
            data["jobs"].extend(added)
        return added
