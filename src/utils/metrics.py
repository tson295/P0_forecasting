"""Frozen metric families: RMSE, MAE, standard R2, RMSE gain vs the E0 baseline."""
import math

import numpy as np
import torch

HORIZON_LABELS = ("1m", "2m", "3m")


def _gain(rmse, rmse_e0):
    """1 - RMSE_model/RMSE_E0: >0 beats E0, 0 equals E0, <0 worse than E0."""
    return [float(1-rmse[i]/rmse_e0[i]) if rmse_e0[i] > 0 else None for i in range(len(rmse))]


class ReturnMetrics:
    """Streaming, sample-weighted metrics in unscaled log-return units."""
    def __init__(self, device):
        self.sums = torch.zeros(4, 3, dtype=torch.float64, device=device)
        self.count = 0

    def update(self, prediction, target):
        p, y = prediction.detach().double(), target.detach().double()
        self.sums[0] += ((p-y)**2).sum(0)
        self.sums[1] += (p-y).abs().sum(0)
        self.sums[2] += y.sum(0)
        self.sums[3] += (y*y).sum(0)
        self.count += len(y)

    def compute(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty dataset")
        s = self.sums.cpu()
        mse = s[0]/self.count
        # E0 predicts return 0 for every horizon, so RMSE_E0 = sqrt(mean(y^2)).
        rmse, rmse_e0 = mse.sqrt(), (s[3]/self.count).sqrt()
        sst = s[3]-s[2]**2/self.count
        return dict(samples=self.count, mse=mse.tolist(), rmse=rmse.tolist(),
                    mae=(s[1]/self.count).tolist(),
                    r2=[float(1-s[0, i]/sst[i]) if sst[i] > 0 else None for i in range(3)],
                    rmse_e0=rmse_e0.tolist(),
                    rmse_gain_vs_e0=_gain(rmse.tolist(), rmse_e0.tolist()),
                    mean_mse=float(mse.mean()))


def price_metrics(origin_mid, target_mid, predicted_mid):
    """Same four families, measured on the mid price in quote currency.

    E0 predicts "the price does not move", so its predicted mid IS the origin mid
    and RMSE_E0 = sqrt(mean((mid_target - mid_origin)^2)) in price units.

    Standard R2 is reported for contract completeness, but in price space it is
    dominated by the level of the price, not by forecast skill: sigma(target mid)
    runs to five figures while every error here is around two, so R2 sits near 1
    for any model including E0. Read rmse_gain_vs_e0 instead.
    """
    origin = np.asarray(origin_mid, dtype=np.float64)
    target = np.asarray(target_mid, dtype=np.float64)
    predicted = np.asarray(predicted_mid, dtype=np.float64)
    if origin.shape != target.shape or target.shape != predicted.shape:
        raise ValueError("origin, target and predicted mids must share a shape")
    if target.ndim != 2 or target.shape[1] != 3 or not len(target):
        raise ValueError("Price metrics need non-empty [N, 3] arrays")
    error, baseline = predicted-target, origin-target
    mse = (error**2).mean(axis=0)
    rmse, rmse_e0 = np.sqrt(mse), np.sqrt((baseline**2).mean(axis=0))
    sst = ((target-target.mean(axis=0))**2).sum(axis=0)
    sse = (error**2).sum(axis=0)
    result = dict(samples=int(len(target)), horizons=list(HORIZON_LABELS), units="quote_currency",
                  mse=mse.tolist(), rmse=rmse.tolist(),
                  mae=np.abs(error).mean(axis=0).tolist(),
                  r2=[float(1-sse[i]/sst[i]) if sst[i] > 0 else None for i in range(3)],
                  rmse_e0=rmse_e0.tolist(),
                  rmse_gain_vs_e0=_gain(rmse.tolist(), rmse_e0.tolist()),
                  mae_e0=np.abs(baseline).mean(axis=0).tolist(),
                  mean_mse=float(mse.mean()))
    if not all(math.isfinite(v) for v in result["mse"]+result["rmse"]+result["mae"]):
        raise FloatingPointError("Nonfinite price metrics")
    return result


def metrics_from_arrays(prediction, target):
    """Exact float64 metrics for the exported prediction tables, same contract."""
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError("Predictions and targets must both be non-empty [N, 3] arrays")
    mse = ((p-y)**2).mean(axis=0)
    rmse, rmse_e0 = np.sqrt(mse), np.sqrt((y*y).mean(axis=0))
    sst = ((y-y.mean(axis=0))**2).sum(axis=0)
    sse = ((p-y)**2).sum(axis=0)
    result = dict(samples=int(len(p)), horizons=list(HORIZON_LABELS),
                  mse=mse.tolist(), rmse=rmse.tolist(),
                  mae=np.abs(p-y).mean(axis=0).tolist(),
                  r2=[float(1-sse[i]/sst[i]) if sst[i] > 0 else None for i in range(3)],
                  rmse_e0=rmse_e0.tolist(),
                  rmse_gain_vs_e0=_gain(rmse.tolist(), rmse_e0.tolist()),
                  mean_mse=float(mse.mean()))
    if not all(math.isfinite(v) for v in result["mse"]+result["rmse"]+result["mae"]):
        raise FloatingPointError("Nonfinite metrics")
    return result
