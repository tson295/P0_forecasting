"""The four official metrics, on the decoded future MID price, per horizon.

  RMSE           sqrt(mean((pred_mid - target_mid)^2))                       USD
  MAE            mean(|pred_mid - target_mid|)                               USD
  R2 gain vs E0  1 - SSE_model/SSE_E0, E0 predicting pred_mid = origin_mid   (WF3 definition)
  DA             directional accuracy, the convention below

Directional-accuracy convention (fixed once, identical for every run):
  pred_delta = pred_mid - origin_mid, true_delta = target_mid - origin_mid.
  Samples whose true_delta is exactly 0 have no direction and are excluded from the
  denominator. On the rest, a hit is sign(pred_delta) == sign(true_delta); a prediction
  of exactly 0 (no call) is therefore a miss. E0 never calls a direction, so its DA is 0
  under this convention; 0.5 is the coin-flip reference.

Nothing else is an official metric. `diagnostics` keeps a few non-official quantities
(expected-value decoding, the E0 reference errors, sample counts) clearly separated.
"""
import math

import numpy as np

OFFICIAL = ("rmse", "mae", "r2_gain_vs_e0", "da")
DA_CONVENTION = ("exclude samples with true_delta == 0; hit iff sign(pred_delta) == sign(true_delta); "
                 "pred_delta == 0 counts as a miss")


def official_metrics(origin_mid, target_mid, pred_mid):
    """One horizon. Arrays are 1-D and aligned; computed in float64."""
    origin = np.asarray(origin_mid, dtype=np.float64)
    target = np.asarray(target_mid, dtype=np.float64)
    pred = np.asarray(pred_mid, dtype=np.float64)
    if not (origin.shape == target.shape == pred.shape) or origin.ndim != 1 or not len(origin):
        raise ValueError("official_metrics needs aligned, non-empty 1-D arrays")
    error, baseline = pred-target, origin-target
    sse, sse_e0 = float((error**2).sum()), float((baseline**2).sum())
    true_delta, pred_delta = target-origin, pred-origin
    moved = true_delta != 0
    hits = np.sign(pred_delta[moved]) == np.sign(true_delta[moved])
    result = dict(rmse=math.sqrt(sse/len(error)), mae=float(np.abs(error).mean()),
                  r2_gain_vs_e0=(1-sse/sse_e0) if sse_e0 > 0 else None,
                  da=float(hits.mean()) if moved.any() else None)
    if not all(math.isfinite(v) for v in (result["rmse"], result["mae"])):
        raise FloatingPointError("Nonfinite official metrics")
    counts = dict(samples=int(len(error)), da_samples=int(moved.sum()),
                  true_zero_excluded=int((~moved).sum()), pred_zero=int((pred_delta == 0).sum()),
                  rmse_e0=math.sqrt(sse_e0/len(error)), mae_e0=float(np.abs(baseline).mean()))
    return result, counts


def official_from_frame(frame, horizon_labels, pred_column="pred_mid"):
    """Recompute the official metrics from an exported prediction table alone."""
    out = {}
    for h in horizon_labels:
        metrics, counts = official_metrics(frame["origin_mid"], frame[f"target_mid_{h}"],
                                           frame[f"{pred_column}_{h}"])
        out[h] = metrics | {"counts": counts}
    return out
