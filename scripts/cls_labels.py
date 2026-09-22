"""Fit and freeze the class definitions from FOLD 1 TRAIN only, then describe every split.

  python scripts/cls_labels.py --csv ~/data/BTC_L10_gate_1y.csv

Writes labeling/<name>.json (the frozen definitions every run loads, never refitted) and
labeling/label_distribution.json (class counts of every fold/split under those frozen
definitions, for the report). The fit functions receive exactly one array: the fold-1
train displacements. Validation, test and folds 2-3 are only ever *assigned*.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.cls import HORIZONS
from src.cls.data import wf3_config
from src.cls.labels import LabelSet, distribution, fit_equal_width, fit_quantile
from src.data.dataset import prepare_data

WF3_TRAINING_COMMIT = "fdbb369946cc07eae9e615c968943036507be129"


def wf3_data_fields(fold):
    """The WF3 data section for a fold; identical across the WF3 model configs."""
    sections = [json.loads((ROOT/f"configs/{m}_wf3_f{fold}.json").read_text())["data"]
                for m in ("ofi_lstm", "moderntcn", "patchtst", "lit", "e0")]
    if any(s != sections[0] for s in sections):
        raise ValueError(f"WF3 data sections differ across models for fold {fold}")
    return sections[0]


def split_deltas(csv, fold):
    """delta[split] = [n, 3] USD displacements for 60/120/180 s, from the WF3 samples."""
    data = prepare_data(wf3_config("moderntcn", wf3_data_fields(fold), csv))
    raw = data.raw
    out, provenance = {}, {}
    for name, ds in data.datasets.items():
        out[name] = raw.mid[ds.target_indices]-raw.mid[ds.origins][:, None]
        provenance[name] = dict(samples=len(ds), rows=list(data.metadata["split_manifest"]["ranges"][name]),
                                origin_index_sha256=hashlib.sha256(ds.origins.tobytes()).hexdigest())
    source = raw.source
    del data
    gc.collect()
    return out, provenance, source


def fitted_on(provenance, source, fold_fields):
    return dict(fold=1, split="train", rows=provenance["train"]["rows"],
                samples=provenance["train"]["samples"],
                origin_index_sha256=provenance["train"]["origin_index_sha256"],
                csv_sha256=source["sha256"], wf3_training_commit=WF3_TRAINING_COMMIT,
                data_config=fold_fields,
                rule="fitted on fold-1 TRAIN displacements only; frozen for every fold and split")


def build_label_sets(train_delta, provenance, source, fold_fields, variants):
    """variants: list of dicts {name, method, ...params}. Only train_delta is ever fitted."""
    record = fitted_on(provenance, source, fold_fields)
    sets = []
    for v in variants:
        specs = []
        for j, h in enumerate(HORIZONS):
            if v["method"] == "equal_width":
                specs.append(fit_equal_width(train_delta[:, j], h, bins=v["bins"],
                                             percentile=v.get("percentile", 99.0),
                                             overflow=v.get("overflow", "edge_plus_half_width"),
                                             fitted_on=record))
            else:
                specs.append(fit_quantile(train_delta[:, j], h, n_classes=v["classes"], fitted_on=record))
        sets.append(LabelSet(v["name"], specs, meta=dict(variant=v, fitted_on=record)))
    return sets


BASELINE_VARIANTS = [
    dict(name="equal_width_k32_p99", method="equal_width", bins=32, percentile=99.0),
    dict(name="quantile_c34", method="quantile", classes=34),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=str(ROOT/"labeling"))
    p.add_argument("--variants", default=None, help="JSON list of extra variants (follow-ups)")
    p.add_argument("--skip-distribution", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = BASELINE_VARIANTS if args.variants is None else json.loads(args.variants)

    deltas = {}
    fold1, provenance, source = split_deltas(args.csv, 1)
    # --- the fit: fold-1 train only ---
    label_sets = build_label_sets(fold1["train"], provenance, source, wf3_data_fields(1), variants)
    for ls in label_sets:
        path = ls.save(out/f"{ls.name}.json")
        print(f"froze {path} sha256={ls.sha256()}")
        for h in HORIZONS:
            s = ls[h]
            print(f"  {h:>3}s classes={s.n_classes} params={ {k: v for k, v in s.params.items() if k in ('q', 'width', 'bins', 'merged_duplicate_edges')} } "
                  f"outer%=({100*s.fit['lower_outer_fraction']:.3f}, {100*s.fit['upper_outer_fraction']:.3f})")
    if args.skip_distribution:
        return
    # --- description only: assign every split of every fold with the frozen definitions ---
    deltas[1] = fold1
    prov = {1: provenance}
    for fold in (2, 3):
        deltas[fold], prov[fold], _ = split_deltas(args.csv, fold)
    report = dict(label_sets={ls.name: ls.sha256() for ls in label_sets}, folds={})
    for fold in (1, 2, 3):
        report["folds"][str(fold)] = {}
        for split, d in deltas[fold].items():
            entry = dict(provenance=prov[fold][split], horizons={})
            for j, h in enumerate(HORIZONS):
                a = np.abs(d[:, j])
                entry["horizons"][str(h)] = dict(
                    rmse_e0=float(np.sqrt((d[:, j]**2).mean())), mean=float(d[:, j].mean()),
                    zero_fraction=float((d[:, j] == 0).mean()),
                    abs_quantiles={str(q): float(np.quantile(a, q)) for q in (.5, .9, .99)},
                    label_sets={ls.name: distribution(ls[h], d[:, j]) for ls in label_sets})
            report["folds"][str(fold)][split] = entry
    (out/"label_distribution.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"wrote {out/'label_distribution.json'}")


if __name__ == "__main__":
    main()
