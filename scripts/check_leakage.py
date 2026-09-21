"""Independent leakage audit for the walk-forward folds.

Re-derives every claim from the raw timestamps rather than trusting the manifest,
so a bug in the splitter cannot certify itself. Exits non-zero on any violation.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.config import Config
from src.data.dataset import prepare_data

MODELS = ("e0", "ofi_lstm", "hfformer", "patchtst", "moderntcn", "lit")


class Audit:
    def __init__(self):
        self.checks = []

    def add(self, name, passed, detail):
        self.checks.append(dict(name=name, passed=bool(passed), detail=detail))
        return bool(passed)

    def failures(self):
        return [c for c in self.checks if not c["passed"]]


def touched_rows(dataset, history_rows):
    """Every raw row a split's samples read: history start through the last target."""
    if not len(dataset):
        return None
    return (int((dataset.origins-history_rows+1).min()), int(dataset.target_indices.max()))


def audit_fold(audit, fold, data, embargo_seconds, config):
    ts = data.raw.timestamps
    history = data.history_rows
    spans = {name: touched_rows(ds, history) for name, ds in data.datasets.items()}
    ranges = data.metadata["split_manifest"]["ranges"]

    for earlier, later in (("train", "validation"), ("validation", "test"), ("train", "test")):
        first, second = spans[earlier], spans[later]
        gap = (int(ts[second[0]])-int(ts[first[1]]))/1e9
        audit.add(f"fold{fold}.{earlier}_to_{later}_gap_seconds", gap > embargo_seconds,
                  dict(gap_seconds=gap, required_greater_than=embargo_seconds,
                       last_row_touched_by=earlier, first_row_touched_by=later,
                       last_row=first[1], first_row=second[0]))

    for earlier, later in (("train", "validation"), ("validation", "test"), ("train", "test")):
        audit.add(f"fold{fold}.{earlier}_{later}_rows_disjoint",
                  spans[earlier][1] < spans[later][0],
                  dict(earlier_last_row=spans[earlier][1], later_first_row=spans[later][0]))

    for name, ds in data.datasets.items():
        lo, hi = ranges[name]
        starts = ds.origins-history+1
        audit.add(f"fold{fold}.{name}_inside_own_split",
                  bool((starts >= lo).all() and (ds.target_indices < hi).all()),
                  dict(min_history_start=int(starts.min()), split_lo=lo,
                       max_target=int(ds.target_indices.max()), split_hi=hi))
        bad = data.raw.bad_prefix
        audit.add(f"fold{fold}.{name}_no_gap_inside_window",
                  bool((bad[ds.target_indices[:, -1]] == bad[starts]).all()),
                  dict(samples=len(ds)))

    origins = {name: set(ds.origins.tolist()) for name, ds in data.datasets.items()}
    for a, b in (("train", "validation"), ("validation", "test"), ("train", "test")):
        audit.add(f"fold{fold}.{a}_{b}_origins_disjoint", not (origins[a] & origins[b]),
                  dict(shared=len(origins[a] & origins[b])))

    scaler = data.metadata["preprocessing"]["standardizer"]
    if scaler is not None:
        # Refit the standardizer on the fold's train rows alone and require an exact match.
        from src.data.preprocessing import Standardizer, features
        x, _ = features(data.raw, config.model, config.data.of_representation, ranges)
        expected = Standardizer.fit(x[:ranges["train"][1]]).to_dict()
        audit.add(f"fold{fold}.standardizer_fitted_on_train_rows_only",
                  expected == scaler,
                  dict(train_hi=ranges["train"][1], matches_refit=expected == scaler))
        del x, expected
        gc.collect()
    else:
        audit.add(f"fold{fold}.standardizer_absent_for_window_local_model", True,
                  dict(model=config.model))
    return {name: dict(rows_touched=spans[name], samples=len(data.datasets[name]),
                       origin_sha=None) for name in data.datasets}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="BTC_L10_gate_1y.csv")
    p.add_argument("--config-template", default="configs/{model}_wf3_f{fold}.json")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--folds", default="1,2,3")
    p.add_argument("--output", default="reports/leakage_audit_wf3.json")
    args = p.parse_args()

    audit = Audit()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    folds = [int(f) for f in args.folds.split(",")]
    test_fingerprint, summary = {}, {}
    for model in models:
        for fold in folds:
            config = Config.load(args.config_template.format(model=model, fold=fold))
            config.data.csv_path = args.csv
            data = prepare_data(config)
            summary[f"{model}.fold{fold}"] = audit_fold(
                audit, f"{fold}.{model}", data, config.data.embargo_seconds, config)
            # Every model and fold must score E0 on exactly the same held-out samples.
            key = (data.datasets["test"].origins.tobytes(),
                   data.datasets["test"].target_indices.tobytes())
            test_fingerprint.setdefault(key, []).append(f"{model}.fold{fold}")
            del data
            gc.collect()
            print(f"audited {model} fold {fold}", flush=True)
    audit.add("test_split_identical_across_every_model_and_fold", len(test_fingerprint) == 1,
              dict(distinct_test_sets=len(test_fingerprint),
                   groups=[v for v in test_fingerprint.values()]))

    failures = audit.failures()
    width = max(len(c["name"]) for c in audit.checks)
    for c in audit.checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']:<{width}}  "
              f"{'' if c['passed'] else json.dumps(c['detail'])[:160]}")
    report = dict(status="fail" if failures else "pass", csv=str(Path(args.csv).resolve()),
                  embargo_seconds_required=180.0, checks=audit.checks,
                  failed=[c["name"] for c in failures], splits=summary)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str)+"\n")
    print(f"\n{len(audit.checks)-len(failures)}/{len(audit.checks)} leakage checks hold; report {out}")
    if failures:
        print("LEAKAGE DETECTED: " + ", ".join(c["name"] for c in failures))
        sys.exit(1)
    print("No leakage detected.")


if __name__ == "__main__":
    main()
