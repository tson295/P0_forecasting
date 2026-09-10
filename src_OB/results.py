"""Raw-price E0 comparisons and tables from completed fold/horizon artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.p0.metrics import gain_pp
from .latency import latency_stats


def gains_vs_e0(metrics, e0):
    # Reuse the existing RMSE gain implementation; export fractions, not pp.
    # Squaring RMSE yields MSE in the SAME price target space and sample set.
    baseline_rmse = e0["RMSE"]
    if baseline_rmse == 0:
        return {"rmse_gain_vs_e0": None, "r2_os_vs_e0": None,
                "e0_gain_status": "undefined_zero_e0_error"}
    return {"rmse_gain_vs_e0": float(gain_pp(metrics["RMSE"], baseline_rmse) / 100),
            "r2_os_vs_e0": float(1 - (metrics["RMSE"] / baseline_rmse) ** 2),
            "e0_gain_status": "defined"}


def atomic_csv(table, path):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def refresh_summaries(output):
    """Serialize writers from fold-parallel Vast jobs, then rebuild completed cells."""
    import fcntl

    output = Path(output)
    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    with (summary / "results.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cells, latency_paths = [], {}
        for path in sorted(output.glob("fold*/*/h*s/metrics.json")):
            if not (path.parent / "completed.json").is_file():
                continue
            metrics = json.loads(path.read_text())
            e0 = metrics["E0"]
            # Also supports older completed cells by using their saved E0 results.
            metrics.update(gains_vs_e0(metrics, e0))
            row = {k: v for k, v in metrics.items() if k != "E0"}
            row.update({f"E0_{k}": v for k, v in e0.items() if k in ("RMSE", "MAE", "R2")})
            cells.append(row)
            key = (row["model"], row["horizon_seconds"])
            latency_paths.setdefault(key, []).append(path.parent / "inference_latency.csv")
        if not cells:
            return
        table = pd.DataFrame(cells).sort_values(["model", "horizon_seconds", "fold"])
        atomic_csv(table, summary / "per_fold_per_horizon.csv")
        groups = []
        for (model, horizon), group in table.groupby(["model", "horizon_seconds"], sort=True):
            row = {"model": model, "horizon_seconds": int(horizon), "n_folds": len(group),
                   "n_predictions": int(group["n"].sum()), "aggregation": "equal_weight_fold_mean"}
            for metric in ("RMSE", "MAE", "R2", "rmse_gain_vs_e0", "r2_os_vs_e0"):
                row[metric] = group[metric].mean()
                row[f"{metric}_min"] = group[metric].min()
                row[f"{metric}_max"] = group[metric].max()
                row[f"{metric}_defined_folds"] = int(group[metric].notna().sum())
            weights = group["n"].to_numpy(float)
            pooled = {"RMSE": float(np.sqrt(np.average(group["RMSE"] ** 2, weights=weights)))}
            e0_pooled = {"RMSE": float(np.sqrt(np.average(group["E0_RMSE"] ** 2, weights=weights)))}
            row["pooled_RMSE"], row["pooled_E0_RMSE"] = pooled["RMSE"], e0_pooled["RMSE"]
            row.update({f"pooled_{k}": v for k, v in gains_vs_e0(pooled, e0_pooled).items()})
            traces = [pd.read_csv(p, usecols=["duration_ms", "n_predictions", "single_origin_sample"])
                      for p in latency_paths[(model, horizon)] if p.is_file()]
            if traces:
                trace = pd.concat(traces, ignore_index=True)
                stats = latency_stats(trace.duration_ms, trace.n_predictions, trace.single_origin_sample)
                # There is no single global "first call" across independent jobs.
                stats.pop("inference_first_call_ms")
                row.update(stats)
            groups.append(row)
        atomic_csv(pd.DataFrame(groups), summary / "by_model_horizon.csv")
