"""Leakage audit for the classification experiment: the WF3 audit, extended.

  python scripts/cls_audit.py --csv ~/data/BTC_L10_gate_1y.csv [--runs-root runs/cls]

Re-derives every claim from the raw file instead of trusting any artifact:

  1  chronological split unchanged (block rows and sample counts = the WF3 run's)
  2  test split identical for every fold, feature set and run
  3  no history window crosses a forbidden split boundary (WF3 audit_fold, hardened floor)
  4  targets isolated: purge + embargo separation > max(embargo, longest horizon)
  5  train-only normalization: each fold's standardizer refitted from its train rows
  6  p99 (equal-width range) fitted ONLY on fold-1 train: refit from raw -> exact match,
     and refitting on any other split gives a different value (the check has power)
  7  quantile edges fitted ONLY on fold-1 train (same test)
  8  class boundaries never use validation/test (6, 7 + recorded provenance)
  9  class representatives never use validation/test (refit -> exact match)
  10 every run of a label set uses byte-identical frozen definitions, in every fold
  11 prediction/test sample fingerprints match across all runs (validation per fold)
  plus, per run: saved split manifest and standardizer match the refit, exported true
  classes and displacements match the raw file, official metrics recompute from the
  exported table, and the best epoch is the validation-CE argmin (no test in selection).

Exits non-zero on any violation. If this fails, no result may be treated as valid.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.cls import HORIZONS
from src.cls.data import FEATURE_MODEL, fingerprint, wf3_config
from src.cls.labels import LabelSet
from src.cls.metrics import official_metrics
from src.data.dataset import prepare_data
from src.data.preprocessing import Standardizer, features

spec = importlib.util.spec_from_file_location("check_leakage", ROOT/"scripts/check_leakage.py")
wf3_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf3_audit)

EXPECTED = dict(csv_sha256="6d8f82fbf3d0d0078b22c44cf5a06c3d82259ca16c71b16bde9b6632a8fbe6df",
                rows_in_file=3139606, rows_used=3139597, duplicate_rows_dropped=9,
                first="2025-09-17T00:00:00+00:00", last="2026-09-16T23:59:50+00:00",
                block_boundary_rows=[0, 666900, 1333126, 1998587, 2667637],
                samples={1: (333177, 332849, 235848), 2: (666059, 332433, 235848),
                         3: (998525, 334350, 235848)})
TAG = {60: "60s", 120: "120s", 180: "180s"}


class Audit(wf3_audit.Audit):
    def add(self, name, passed, detail=None):
        return super().add(name, passed, detail or {})


def refit_label_set(frozen, train_delta):
    """Refit a frozen label set's own variant from a displacement array."""
    from src.cls.labels import fit_equal_width, fit_quantile
    v = frozen.meta["variant"]
    specs = []
    for j, h in enumerate(HORIZONS):
        if v["method"] == "equal_width":
            specs.append(fit_equal_width(train_delta[:, j], h, bins=v["bins"], percentile=v.get("percentile", 99.0),
                                         overflow=v.get("overflow", "edge_plus_half_width")))
        else:
            specs.append(fit_quantile(train_delta[:, j], h, n_classes=v["classes"]))
    return LabelSet(frozen.name, specs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--runs-root", default=str(ROOT/"runs/cls"))
    p.add_argument("--labeling", default=str(ROOT/"labeling"))
    p.add_argument("--output", default=str(ROOT/"audits/leakage_audit_classification.json"))
    p.add_argument("--data-only", action="store_true", help="Pre-flight: skip the per-run checks")
    args = p.parse_args()
    audit = Audit()
    labels = {path.stem: LabelSet.load(path) for path in sorted(Path(args.labeling).glob("*.json"))
              if path.name != "label_distribution.json"}

    canonical = {}   # fold -> split -> dict(origins, targets, fingerprint)
    standardizers = {}  # (feature model, fold) -> dict
    delta = {}       # fold -> split -> [n, 3]
    mids = None
    for fold in (1, 2, 3):
        for arch in ("ofi_lstm", "moderntcn", "transformer"):
            fields = json.loads((ROOT/f"configs/moderntcn_wf3_f{fold}.json").read_text())["data"]
            config = wf3_config(arch, fields, args.csv)
            data = prepare_data(config)
            raw, man = data.raw, data.metadata["split_manifest"]
            tag = f"fold{fold}.{arch}"
            if fold == 1 and arch == "ofi_lstm":
                rep = raw.source["row_repairs"]
                audit.add("dataset.sha256", raw.source["sha256"] == EXPECTED["csv_sha256"], dict(sha256=raw.source["sha256"]))
                audit.add("dataset.rows_in_file", rep["rows_in_file"] == EXPECTED["rows_in_file"], rep)
                audit.add("dataset.rows_after_repair", rep["rows_used"] == EXPECTED["rows_used"]
                          and rep["duplicate_rows_dropped"] == EXPECTED["duplicate_rows_dropped"]
                          and rep["duplicate_timestamp_policy"] == "keep_first" and rep["sorted_by_timestamp"], rep)
                span = [pd.Timestamp(int(raw.timestamps[i]), unit="ns", tz="UTC").isoformat() for i in (0, -1)]
                audit.add("dataset.date_range", span == [EXPECTED["first"], EXPECTED["last"]], dict(span=span))
                mids = raw.mid
            # 1 chronological split unchanged
            audit.add(f"{tag}.split_block_rows_equal_wf3",
                      man["block_boundary_rows"][:5] == EXPECTED["block_boundary_rows"],
                      dict(rows=man["block_boundary_rows"]))
            counts = tuple(man["sample_counts"][s] for s in ("train", "validation", "test"))
            audit.add(f"{tag}.sample_counts_equal_wf3", counts == EXPECTED["samples"][fold], dict(counts=counts))
            # 3, 4, 5: the WF3 audit (window containment, gaps, disjointness, separation floor,
            # standardizer refit from train rows) on this feature set and fold
            wf3_audit.audit_fold(audit, f"{fold}.{arch}", data, config.data.embargo_seconds, config)
            standardizers[(arch, fold)] = data.metadata["preprocessing"]["standardizer"]
            for split, ds in data.datasets.items():
                entry = dict(origins=ds.origins.copy(), targets=ds.target_indices.copy(),
                             fingerprint=fingerprint(ds.origins, ds.target_indices))
                if arch == "ofi_lstm":
                    canonical.setdefault(fold, {})[split] = entry
                    delta.setdefault(fold, {})[split] = raw.mid[ds.target_indices]-raw.mid[ds.origins][:, None]
                else:
                    audit.add(f"{tag}.{split}_samples_identical_across_feature_sets",
                              entry["fingerprint"] == canonical[fold][split]["fingerprint"])
            del data, raw
            gc.collect()
            print(f"audited data {tag}", flush=True)
    # 2 test identical across folds
    test_prints = {canonical[f]["test"]["fingerprint"] for f in (1, 2, 3)}
    audit.add("test_split_identical_across_folds", len(test_prints) == 1, dict(fingerprints=sorted(test_prints)))
    test_fingerprint = canonical[1]["test"]["fingerprint"]

    # 6-9 label definitions: refit from fold-1 train, and prove the check has power
    f1_train = delta[1]["train"]
    f1_origins_sha = hashlib.sha256(canonical[1]["train"]["origins"].tobytes()).hexdigest()
    for name, frozen in labels.items():
        refit = refit_label_set(frozen, f1_train)
        for h in HORIZONS:
            a, b = frozen[h], refit[h]
            audit.add(f"labels.{name}.{h}s.edges_refit_from_f1_train_exact", a.edges == b.edges)
            audit.add(f"labels.{name}.{h}s.representatives_refit_from_f1_train_exact",
                      a.representatives == b.representatives)
            audit.add(f"labels.{name}.{h}s.n_classes_refit_exact", a.n_classes == b.n_classes)
            if a.method == "equal_width":
                audit.add(f"labels.{name}.{h}s.q_is_f1_train_percentile", a.params["q"] == b.params["q"],
                          dict(q=a.params["q"]))
            fit = a.fit["fitted_on"]
            audit.add(f"labels.{name}.{h}s.provenance_fold1_train",
                      fit.get("fold") == 1 and fit.get("split") == "train"
                      and fit.get("origin_index_sha256") == f1_origins_sha
                      and fit.get("samples") == len(f1_train),
                      dict(fitted_on={k: fit.get(k) for k in ("fold", "split", "samples", "origin_index_sha256")}))
            audit.add(f"labels.{name}.{h}s.fit_delta_sha256_is_f1_train",
                      a.fit["delta_sha256"] == hashlib.sha256(np.ascontiguousarray(f1_train[:, HORIZONS.index(h)]).tobytes()).hexdigest())
        # Power: a definition refitted on any other split must differ.
        for fold, split in ((1, "validation"), (1, "test"), (2, "train"), (3, "train"), (3, "validation")):
            other = refit_label_set(frozen, delta[fold][split])
            differs = all(other[h].edges != frozen[h].edges for h in HORIZONS)
            audit.add(f"labels.{name}.refit_on_f{fold}_{split}_differs", differs)

    # per-run checks
    runs = sorted(Path(args.runs_root).rglob("run_summary.json")) if not args.data_only else []
    runs = [r for r in runs if "benchmark_work" not in str(r)]
    by_definition = {}
    summary_runs = []
    for summary_path in runs:
        run = summary_path.parent
        s = json.loads(summary_path.read_text())
        rid = s["job_id"]
        fold = s["fold"]
        cfg = json.loads((run/"run_config.json").read_text())
        frozen_name = Path(cfg["label_set_path"]).stem
        frozen = labels[frozen_name]
        used = LabelSet.load(run/"label_set.json")
        used_best = LabelSet.load(run/"best/label_set.json")
        by_definition.setdefault(frozen_name, set()).update({used.sha256(), used_best.sha256()})
        audit.add(f"run.{rid}.label_definition_is_frozen_file",
                  used.sha256() == frozen.sha256() == used_best.sha256() == s["label_definition_sha256"])
        manifest = json.loads((run/"best/split_manifest.json").read_text())
        audit.add(f"run.{rid}.split_manifest_matches_refit",
                  manifest["origin_index_sha256"]["train"] == hashlib.sha256(canonical[fold]["train"]["origins"].tobytes()).hexdigest()
                  and manifest["origin_index_sha256"]["validation"] == hashlib.sha256(canonical[fold]["validation"]["origins"].tobytes()).hexdigest()
                  and manifest["origin_index_sha256"]["test"] == hashlib.sha256(canonical[fold]["test"]["origins"].tobytes()).hexdigest())
        prep = json.loads((run/"best/preprocessing.json").read_text())
        audit.add(f"run.{rid}.standardizer_is_train_only_refit",
                  prep["standardizer"] == standardizers[(s["arch"], fold)])
        history = [json.loads(l) for l in (run/"training_history.jsonl").read_text().splitlines() if l.strip()]
        ces = [r["validation"]["mean_ce"] for r in history]
        audit.add(f"run.{rid}.best_epoch_is_validation_ce_argmin",
                  int(np.argmin(ces)) == s["best_epoch"] and not any("test" in r for r in history))
        for split in ("validation", "test"):
            frame = pd.read_csv(run/f"{split}_predictions.csv.gz")
            origins = frame["origin_index"].to_numpy(np.int64)
            ref = canonical[fold][split]
            cols = [ref["targets"][:, HORIZONS.index(h)] for h in s["horizons"]]
            same = np.array_equal(origins, ref["origins"]) and all(
                np.array_equal(frame[f"target_index_{TAG[h]}"].to_numpy(np.int64), c) for h, c in zip(s["horizons"], cols))
            audit.add(f"run.{rid}.{split}_samples_match_canonical", same)
            if split == "test":
                audit.add(f"run.{rid}.test_fingerprint_is_shared", s["test_sample_fingerprint"] == test_fingerprint)
            else:
                audit.add(f"run.{rid}.validation_fingerprint_is_fold_{fold}",
                          s["validation_sample_fingerprint"] == ref["fingerprint"])
            metrics = json.loads((run/f"{split}_metrics.json").read_text())["official_metrics"]
            ok_labels, ok_metrics = True, True
            for h in s["horizons"]:
                t = TAG[h]
                true_delta = mids[frame[f"target_index_{t}"].to_numpy(np.int64)]-mids[origins]
                ok_labels &= np.array_equal(frame[f"true_delta_{t}"].to_numpy(), true_delta)
                ok_labels &= np.array_equal(frame[f"true_class_{t}"].to_numpy(np.int64), frozen[h].assign(true_delta))
                ok_labels &= np.array_equal(frame[f"pred_delta_{t}"].to_numpy(),
                                            frozen[h].decode(frame[f"pred_class_{t}"].to_numpy(np.int64)))
                recomputed, _ = official_metrics(frame["origin_mid"], frame[f"target_mid_{t}"], frame[f"pred_mid_{t}"])
                ok_metrics &= all(abs(recomputed[k]-metrics[t][k]) <= 1e-9*max(1.0, abs(metrics[t][k])) for k in recomputed)
            audit.add(f"run.{rid}.{split}_true_classes_and_decoding_match_raw_and_frozen_labels", bool(ok_labels))
            audit.add(f"run.{rid}.{split}_official_metrics_recompute_from_table", bool(ok_metrics))
        summary_runs.append(rid)
        print(f"audited run {rid}", flush=True)
    for name, shas in by_definition.items():
        audit.add(f"labels.{name}.identical_definition_in_every_run_and_fold", len(shas) == 1, dict(shas=sorted(shas)))

    failures = audit.failures()
    report = dict(status="fail" if failures else "pass", csv=str(Path(args.csv).resolve()),
                  checks_total=len(audit.checks), checks_passed=len(audit.checks)-len(failures),
                  failed=[c["name"] for c in failures], runs_audited=summary_runs,
                  label_sets={n: ls.sha256() for n, ls in labels.items()},
                  test_sample_fingerprint=test_fingerprint,
                  embargo_floor="max(embargo_seconds, longest horizon) (WF3 124b1e9 hardening)",
                  checks=audit.checks)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str)+"\n")
    print(f"\n{report['checks_passed']}/{report['checks_total']} checks hold over {len(summary_runs)} runs; report {out}")
    if failures:
        print("AUDIT FAILED: "+", ".join(c["name"] for c in failures[:50]))
        sys.exit(1)
    print("No leakage detected.")


if __name__ == "__main__":
    main()
