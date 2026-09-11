"""Batched rolling-origin inference for the pinned AutoTS 1.0.4 WR/MR classes.

Origins are independent columns while generating MR rolling statistics, then
independent rows for the fitted estimator. Only forecast steps are recursive.
Unsupported transformations fail explicitly instead of mixing origin histories.
"""
from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src_OB.gpu import GPUOnlyError, GPURegressor

_PATCH_LOCK = threading.RLock()


class AutoTSGPURegressor:
    """Wrap native estimators, including LightGBM's multi-output wrapper."""
    def __init__(self, estimator, family):
        self.estimator, self.family = estimator, family
        self.fitted = []

    def fit(self, x, y, **kwargs):
        from sklearn.base import clone
        from sklearn.multioutput import MultiOutputRegressor

        self.fitted = []
        if isinstance(self.estimator, MultiOutputRegressor):
            target = np.asarray(y)
            if target.ndim != 2:
                raise ValueError("Multi-output AutoTS target must have two dimensions")
            for col in range(target.shape[1]):
                self.fitted.append(GPURegressor(clone(self.estimator.estimator), self.family).fit(x, target[:, col], **kwargs))
            self.multioutput = True
        else:
            self.fitted.append(GPURegressor(self.estimator, self.family).fit(x, y, **kwargs))
            self.multioutput = False
        return self

    def predict(self, x, **kwargs):
        if self.multioutput:
            return np.column_stack([m.predict(x, **kwargs) for m in self.fitted])
        return self.fitted[0].predict(x, **kwargs)


@contextmanager
def gpu_regressors():
    """Guard real fits in both direct models and native AutoTS template search."""
    import autots.models.sklearn as native

    with _PATCH_LOCK:
        original = native.retrieve_regressor

        def retrieve(regression_model, *args, **kwargs):
            name = regression_model["model"]
            if name not in ("LightGBM", "xgboost"):
                raise GPUOnlyError(f"AutoTS backend outside GPU allowlist: {name}")
            return AutoTSGPURegressor(original(regression_model, *args, **kwargs),
                                      "lgbm" if name == "LightGBM" else "xgb")

        native.retrieve_regressor = retrieve
        try:
            yield
        finally:
            native.retrieve_regressor = original


def covariates(seq):
    if seq.cov is None:
        return np.empty((len(seq.idx), 0), dtype=np.float64)
    x = np.asarray(seq.cov[seq.idx], dtype=np.float64).copy()
    for col, alt in (seq.perm or {}).items():
        x[:, col] = seq.cov[np.asarray(alt, dtype=np.int64), col]
    return x


def _guard(m, kind):
    from importlib.metadata import version

    if version("autots") != "1.0.4":
        raise RuntimeError("Batched AutoTS adapter requires the pinned autots==1.0.4")
    if len(m.column_names) != 1 or m.regression_type not in ("User", "user"):
        raise ValueError("Batched adapter expects one r1 target and User regressors")
    if m.datepart_method not in (None, "None", "none"):
        raise ValueError("Batched adapter does not accept datepart features")
    if kind == "wr":
        if m.output_dim != "forecast_length" or m.input_dim not in ("univariate", "multivariate"):
            raise ValueError("WR batch requires direct forecast_length output")
    else:
        if (any(getattr(m, name, None) for name in
                ("holiday", "transformation_dict", "cointegration", "polynomial_degree", "probabilistic"))
                or m.regressor_per_series_train is not None or m.static_regressor is not None):
            raise ValueError("MR batch does not support transforms/cross-series/interval features")


def predict_wr(m, seq, batch_size, timings=None):
    import torch

    _guard(m, "wr")
    horizon, width = int(m.forecast_length), int(m.window_size)
    if len(seq.idx) and np.min(seq.idx) < width - 1:
        raise ValueError("Insufficient WR context")
    out = np.empty((len(seq.idx), horizon), dtype=np.float64)
    future = covariates(seq)
    for start in range(0, len(seq.idx), batch_size):
        torch.cuda.synchronize()
        started = time.perf_counter()
        idx = seq.idx[start:start + batch_size]
        positions = idx[:, None] - np.arange(width - 1, -1, -1)
        if not np.all(seq.ts[positions] == seq.ts[idx, None] - np.arange(width - 1, -1, -1) * 60):
            raise ValueError("WR context crosses a timestamp gap")
        windows = np.asarray(seq.r1[positions], dtype=np.float64)
        if m.normalize_window:
            windows = windows / windows.sum(axis=1, keepdims=True)
        x = np.column_stack((windows, future[start:start + batch_size]))
        if m.scale:
            x = m.scaler.transform(x)
        if m.fourier_encoding_components is not None:
            x = m.fourier_encoder.transform(x)
        # Native WR casts its assembled DataFrame through float32 before predict.
        pred = np.asarray(m.model.predict(x.astype(np.float32).astype(float)))
        if pred.shape != (len(idx), horizon) or not np.isfinite(pred).all():
            raise ValueError("Invalid batched WR prediction")
        out[start:start + len(idx)] = pred
        torch.cuda.synchronize()
        if timings is not None:
            timings.append({"start": start, "origins": len(idx), "steps": horizon,
                            "seconds": time.perf_counter() - started})
    return out


def predict_mr(m, seq, batch_size, tail_bars, timings=None):
    from autots.models.sklearn import rolling_x_regressor
    import torch

    _guard(m, "mr")
    horizon = int(m.forecast_length)
    width = min(int(tail_bars), int(m.min_threshold))
    if len(seq.idx) and np.min(seq.idx) < width - 1:
        raise ValueError("Insufficient MR context")
    params = {name: getattr(m, name) for name in (
        "mean_rolling_periods", "macd_periods", "std_rolling_periods", "max_rolling_periods",
        "min_rolling_periods", "ewm_var_alpha", "quantile90_rolling_periods", "quantile10_rolling_periods",
        "additional_lag_periods", "ewm_alpha", "abs_energy", "rolling_autocorr_periods", "nonzero_last_n",
        "window", "rolling_skew_periods", "diff_periods", "rolling_range_periods")}
    future = covariates(seq)
    out = np.empty((len(seq.idx), horizon), dtype=np.float64)
    for start in range(0, len(seq.idx), batch_size):
        torch.cuda.synchronize()
        started = time.perf_counter()
        idx = seq.idx[start:start + batch_size]
        count = len(idx)
        positions = idx[:, None] - np.arange(width - 1, -1, -1)
        if not np.all(seq.ts[positions] == seq.ts[idx, None] - np.arange(width - 1, -1, -1) * 60):
            raise ValueError("MR context crosses a timestamp gap")
        histories = np.asarray(seq.r1[positions], dtype=float).T
        # Time-dependent features are prohibited above. This aligned index only
        # gives pandas a minute frequency; each column contains its own history.
        index = pd.to_datetime(seq.ts[positions[0]], unit="s")
        for step in range(horizon):
            frame = pd.DataFrame(histories, index=index)
            features = rolling_x_regressor(frame, **params)
            # Every enabled feature contributes count columns, in feature-major
            # order. No polynomial/cointegration/date features may mix columns.
            last = features.iloc[-1].to_numpy(dtype=float)
            if len(last) % count:
                raise ValueError("AutoTS rolling feature layout changed")
            x = np.column_stack((last.reshape(-1, count).T, future[start:start + count]))
            if m.series_hash:
                hashed = int(hashlib.sha256(str(m.column_names[0]).encode()).hexdigest(), 16) % 10**16
                x = np.column_stack((x, np.full(count, hashed)))
            if m.scale_full_X:
                if m.scaler_mean is None:
                    raise ValueError("MR scaler must be fitted on training data")
                x = (x - np.asarray(m.scaler_mean)) / np.asarray(m.scaler_std)
            x = np.clip(x, -1e12, 1e12)
            mask = getattr(m, "_nonzero_var_mask", None)
            if mask is not None:
                x = x[:, mask]
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            pred = np.asarray(m.model.predict(x), dtype=float).reshape(-1)
            if pred.shape != (count,) or not np.isfinite(pred).all():
                raise ValueError("Invalid batched MR prediction")
            out[start:start + count, step] = pred
            # Match native MR: append predictions, do not trim history between steps.
            histories = np.vstack((histories, pred[None, :]))
            index = index.append(pd.DatetimeIndex([index[-1] + pd.Timedelta(minutes=1)]))
        torch.cuda.synchronize()
        if timings is not None:
            timings.append({"start": start, "origins": count, "steps": horizon,
                            "seconds": time.perf_counter() - started})
    return out
