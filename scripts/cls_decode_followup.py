"""Follow-ups that change ONLY the decoding rule of already-trained models.

  python scripts/cls_decode_followup.py

For every completed run, the saved class probabilities (<split>_probs.npz, float16) are
decoded with alternative rules; the model, the classes and the samples are unchanged, so
each rule is one controlled change against the official argmax decoding:

  argmax              official: representative of the most probable class
  expected            sum_k p_k * representative_k (the prompt's secondary diagnostic)
  overflow_median     equal-width only: argmax, but the two overflow classes decode to the
                      fold-1-train median displacement of their own samples instead of
                      -q - w/2 / +q + w/2 (labeling/equal_width_k32_p99_ovfmedian.json)

Learning-free references from the run's own TRAIN class frequencies (train_metrics.json):
  prior_argmax        always the most frequent train class, decoded (a constant prediction)
  prior_expected      sum_k prior_k * representative_k (a constant, near 0)
and the cross-entropy of that prior on the split (`prior_ce`) next to the model's CE, so
"did the classifier learn anything beyond the marginal" is answered per run.

Writes reports/followups/decoding_long.csv (official four metrics per run x split x
horizon x rule) and prints a summary. Validation and test are scored separately; any
choice between rules must be read off validation.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.cls.labels import LabelSet
from src.cls.metrics import official_metrics

TAGS = {60: "60s", 120: "120s", 180: "180s"}


def main():
    ovf = LabelSet.load(ROOT/"labeling/equal_width_k32_p99_ovfmedian.json")
    rows = []
    for summary_path in sorted((ROOT/"runs/cls").rglob("run_summary.json")):
        run = summary_path.parent
        if "benchmark_work" in run.parts:
            continue
        s = json.loads(summary_path.read_text())
        cfg = json.loads((run/"run_config.json").read_text())
        ls = LabelSet.load(run/"label_set.json")
        train_diag = json.loads((run/"train_metrics.json").read_text())["diagnostics_not_official"]
        for split in ("validation", "test"):
            split_diag = json.loads((run/f"{split}_metrics.json").read_text())["diagnostics_not_official"]
            frame = pd.read_csv(run/f"{split}_predictions.csv.gz", float_precision="round_trip")
            probs = np.load(run/f"{split}_probs.npz")
            origin = frame["origin_mid"].to_numpy()
            for h in s["horizons"]:
                tag = TAGS[h]
                target = frame[f"target_mid_{tag}"].to_numpy()
                p = probs[tag].astype(np.float64)
                p /= p.sum(1, keepdims=True)
                reps = np.asarray(ls[h].representatives)
                cls = frame[f"pred_class_{tag}"].to_numpy(np.int64)
                hist = np.asarray(train_diag[tag]["true_class_histogram"], dtype=np.float64)
                prior = hist/hist.sum()
                true_cls = frame[f"true_class_{tag}"].to_numpy(np.int64)
                prior_ce = float(-np.log(np.clip(prior[true_cls], 1e-12, None)).mean())
                model_ce = split_diag[tag]["cross_entropy"]
                decoded = dict(argmax=reps[cls], expected=p @ reps,
                               prior_argmax=np.full(len(frame), reps[int(prior.argmax())]),
                               prior_expected=np.full(len(frame), float(prior @ reps)))
                if ls[h].method == "equal_width" and ls.name == "equal_width_k32_p99":
                    if ovf[h].edges != ls[h].edges:
                        raise AssertionError("overflow variant must share the edges")
                    decoded["overflow_median"] = np.asarray(ovf[h].representatives)[cls]
                for rule, delta in decoded.items():
                    m, c = official_metrics(origin, target, origin+delta)
                    rows.append(dict(job_id=s["job_id"], method=s["method"], label_set=ls.name, arch=s["arch"],
                                     formulation=cfg["formulation"], experiment=cfg.get("experiment") or "",
                                     variant=cfg.get("variant", "baseline"),
                                     fold=s["fold"], horizon=tag, split=split, rule=rule, **m,
                                     mean_abs_decoded=float(np.abs(delta).mean()),
                                     share_decoded_zero=float((delta == 0).mean()),
                                     model_ce=model_ce, prior_ce=prior_ce))
        print("decoded", s["job_id"], flush=True)
    df = pd.DataFrame(rows)
    out = ROOT/"reports/followups"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"decoding_long.csv", index=False)
    base = df[df.experiment == ""]
    summary = base.groupby(["split", "rule", "method", "formulation", "horizon"])[["r2_gain_vs_e0", "da"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
