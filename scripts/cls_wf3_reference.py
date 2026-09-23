"""Score the WF3 regression runs with the classification benchmark's four metrics.

  python scripts/cls_wf3_reference.py --csv ~/data/BTC_L10_gate_1y.csv

Downloads the WF3 test predictions (Tson29/Pretrain_Model_WF3, <model>_wf3_f<fold>) for
OFI-LSTM and ModernTCN, checks that their test samples are exactly this experiment's
test samples, and recomputes RMSE / MAE / R2 gain vs E0 / DA with src/cls/metrics.py
(WF3 never reported DA). Writes reports/wf3_regression_reference.json.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from src.cls.data import fingerprint, wf3_config
from src.cls.metrics import official_metrics
from src.data.dataset import prepare_data

REPO = "Tson29/Pretrain_Model_WF3"
TAGS = {"1m": 60, "2m": 120, "3m": 180}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=str(ROOT/"reports/wf3_regression_reference.json"))
    args = p.parse_args()
    fields = json.loads((ROOT/"configs/moderntcn_wf3_f1.json").read_text())["data"]
    data = prepare_data(wf3_config("moderntcn", fields, args.csv))
    test = data.datasets["test"]
    ours = fingerprint(test.origins, test.target_indices)
    out = dict(source=REPO, test_sample_fingerprint=ours, runs={})
    for model in ("ofi_lstm", "moderntcn"):
        for fold in (1, 2, 3):
            run = f"{model}_wf3_f{fold}"
            path = hf_hub_download(REPO, f"{model}/{run}/artifacts/test_predictions.csv.gz")
            frame = pd.read_csv(path, float_precision="round_trip")
            targets = np.stack([frame[f"target_index_{t}"].to_numpy(np.int64) for t in TAGS], 1)
            same = fingerprint(frame["origin_index"].to_numpy(np.int64), targets) == ours
            if not same:
                raise ValueError(f"{run}: WF3 test samples differ from this experiment's test samples")
            metrics = {}
            for tag, h in TAGS.items():
                m, c = official_metrics(frame["origin_mid"], frame[f"target_mid_{tag}"], frame[f"pred_mid_{tag}"])
                metrics[f"{h}s"] = m | {"counts": c}
            out["runs"][run] = dict(model=model, fold=fold, same_test_samples=same, test=metrics,
                                    formulation="WF3 regression: MSE on log return, pred_mid = origin_mid*exp(pred)")
            print(run, {h: round(v["r2_gain_vs_e0"], 6) for h, v in metrics.items()},
                  {h: round(v["da"], 4) for h, v in metrics.items()}, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    main()
