"""§7 log: experiments/log.csv (mỗi run một dòng) + runs/<exp_id>/ (config, prediction)."""
from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import numpy as np

LOG_FIELDS = ["exp_id", "timestamp", "step", "model", "dataset_label", "config_hash", "seed", "colset", "n_cols", "rounds", "base",
              "MedianGain", "WinRate", "P10Gain", "WorstGain", "rmse_cells", "mae_cells", "e0_cells", "gain_cells", "train_device",
              "decision", "note"]
CHAMPION_FIELDS = ["exp_id", "timestamp", "model", "win", "n_ext", "ext_cols", "members", "weighting", "MedianGain_vs_E0",
                   "champion_before", "MedianGain_vs_champion", "WinRate", "P10Gain", "WorstGain", "eps_champion", "decision", "champion_after",
                   "rmse_mean_win", "rmse_mean_champion", "gain_cells", "rmse_h1", "rmse_h2", "rmse_h3", "mae_h1", "mae_h2", "mae_h3",
                   "champ_rmse_h1", "champ_rmse_h2", "champ_rmse_h3", "latency_p95_ms", "latency_p99_ms", "latency_max_ms",
                   "train_device", "predict_device"]
LATENCY_FIELDS = ["timestamp", "split", "model", "h", "n", "p95_ms", "p99_ms", "max_ms", "mean_ms", "shared", "train_device", "predict_device",
                  "lib_version", "threads"]


def new_exp_id(step: str, model: str, extra: str = "") -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{step}_{model}{('_' + extra) if extra else ''}"


_APPEND_LOCK = threading.Lock()  # nhiều nhánh model chạy song song (orchestrate) cùng ghi log.csv / champion_log.csv


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    """Append một dòng với schema CỐ ĐỊNH (cột thiếu → rỗng, cột thừa → bỏ) → file luôn đọc được bằng pd.read_csv.

    Có lock: các nhánh song song không bao giờ ghi xen giữa dòng của nhau (§19 race-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with _APPEND_LOCK, open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        r = {k: row.get(k, "") for k in fields}
        if "timestamp" in fields and not r["timestamp"]:
            r["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        w.writerow(r)


def log_run(exp_dir: Path, row: dict) -> None:
    append_csv(exp_dir / "log.csv", LOG_FIELDS, row)


def log_champion(exp_dir: Path, row: dict) -> None:
    append_csv(exp_dir / "champion_log.csv", CHAMPION_FIELDS, row)


def log_latency(exp_dir: Path, lat_df, split: str) -> None:
    """§7.4: gom latency mọi model vào summary/latency_summary.csv (split = VAL | TEST)."""
    for r in lat_df.to_dict("records"):
        append_csv(exp_dir / "summary" / "latency_summary.csv", LATENCY_FIELDS, {**r, "split": split})


def save_run(exp_dir: Path, exp_id: str, payload: dict, preds: list[tuple[np.ndarray, np.ndarray]] | None = None,
             pred_name: str = "pred_val.npz") -> Path:
    """§7: experiments/runs/<exp_id>/run.json (config, colset, số vòng, metric 15 ô) + prediction (idx origin, ŷ log-return) nếu có."""
    d = exp_dir / "runs" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    if preds:
        np.savez_compressed(d / pred_name, **{f"idx_{i}": p[0] for i, p in enumerate(preds)}, **{f"yhat_{i}": p[1] for i, p in enumerate(preds)})
    return d


def load_preds(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    z = np.load(path)
    n = len([k for k in z.files if k.startswith("idx_")])
    return [(z[f"idx_{i}"], z[f"yhat_{i}"]) for i in range(n)]


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)
