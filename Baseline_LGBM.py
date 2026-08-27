"""
Standalone OHLCV-only LightGBM baseline.

Input requirements
------------------
A pandas DataFrame with:
    timestamp, Open, High, Low, Close, Volume

No VWAP is required.

The module:
1) builds all causal technical features from OHLCV only;
2) builds lagged tabular features;
3) builds h=1,2,3 minute log-return targets;
4) trains THREE independent LightGBM regressors;
5) requires GPU LightGBM and optionally enforces NVIDIA P100.

The default feature set mirrors the current V2 feature engineering:
- returns / candle geometry
- volume / relative volume / volume z-score
- realized volatility
- RSI
- MACD histogram + acceleration
- EMA gap
- HMA slope
- sign-flip rate
- time encodings
- 8-minute coarse features

No future information is used.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import lightgbm as lgb


HORIZONS = (1, 2, 3)

FINE_FEATURE_NAMES = (
    "return1",
    "return5",
    "candle_return",
    "log_range",
    "body_over_range",
    "close_position",
    "upper_wick",
    "lower_wick",
    "log_volume_delta",
    "relative_log_volume60",
    "volume_z60",
    "log_rv5_rv60",
    "minute_mod5_sin",
    "minute_mod5_cos",
    "time_of_day_sin",
    "time_of_day_cos",
    "rsi15_centered",
    "macd_hist_5_20_7_volnorm",
    "macd_hist_acceleration",
    "ema_gap_8_32_volnorm",
    "hma_slope16_volnorm",
    "sign_flip_rate8",
)

COARSE_FEATURE_NAMES = (
    "return8",
    "return32",
    "body8",
    "range8",
    "close_position8",
    "log_volume8_delta",
    "relative_log_volume8_64",
    "rv8",
    "rv64",
    "log_rv8_rv64",
    "time_of_day_sin",
    "time_of_day_cos",
    "rsi64_centered",
    "macd_hist_16_64_16_volnorm",
    "ema_gap_32_128_volnorm",
    "sign_flip_rate32",
)

# Causal row offsets from the prediction origin.
FINE_LAGS = (-63, -32, -16, -8, -4, -2, -1, 0)
COARSE_LAGS = (-504, -256, -128, -64, -32, -16, -8, 0)


@dataclass(frozen=True)
class LGBMConfig:
    require_p100: bool = True

    device_type: str = "gpu"
    gpu_use_dp: bool = False
    max_bin: int = 63

    objective: str = "huber"
    alpha: float = 0.90

    n_estimators: int = 1200
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 80

    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.85

    reg_alpha: float = 1e-4
    reg_lambda: float = 1e-2

    early_stopping_rounds: int = 80
    seed: int = 8586
    n_jobs: int = -1


@dataclass
class TargetTransform:
    mean: np.ndarray
    scale: np.ndarray
    volatility_floor: float

    @classmethod
    def fit(
        cls,
        y: np.ndarray,
        rv60: np.ndarray,
        train_idx: np.ndarray,
    ) -> "TargetTransform":
        train_rv = rv60[train_idx]
        floor = float(np.quantile(train_rv, 0.01))
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("Invalid RV60 floor")

        denom = np.maximum(train_rv, floor)[:, None]
        denom *= np.sqrt(np.asarray(HORIZONS, np.float32))[None, :]

        normalized = y[train_idx] / denom
        mean = np.mean(normalized, axis=0).astype(np.float32)
        scale = np.std(normalized, axis=0).astype(np.float32)
        scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)

        return cls(mean, scale, floor)

    def encode(
        self,
        y: np.ndarray,
        rv60: np.ndarray,
        idx: np.ndarray,
    ) -> np.ndarray:
        denom = np.maximum(rv60[idx], self.volatility_floor)[:, None]
        denom *= np.sqrt(np.asarray(HORIZONS, np.float32))[None, :]
        normalized = y[idx] / denom
        return ((normalized - self.mean) / self.scale).astype(np.float32)

    def decode(
        self,
        z: np.ndarray,
        rv60: np.ndarray,
    ) -> np.ndarray:
        normalized = z * self.scale + self.mean
        denom = np.maximum(rv60, self.volatility_floor)[:, None]
        denom *= np.sqrt(np.asarray(HORIZONS, np.float32))[None, :]
        return (normalized * denom).astype(np.float32)


@dataclass
class FeatureData:
    frame: pd.DataFrame
    fine: np.ndarray
    coarse: np.ndarray
    rv60: np.ndarray
    target: np.ndarray
    eligible: np.ndarray


@dataclass
class LGBMResult:
    models: list
    prediction: np.ndarray
    encoded_prediction: np.ndarray
    feature_names: tuple[str, ...]
    importance: pd.DataFrame
    best_iterations: tuple[int, ...]
    target_transform: TargetTransform


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.astype(float)
    b = b.astype(float)
    out = pd.Series(np.nan, index=a.index, dtype=float)
    mask = (a > 0) & (b > 0)
    out.loc[mask] = np.log(a.loc[mask] / b.loc[mask])
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    EMA that resets after missing-value gaps.
    """
    segment = series.isna().cumsum()
    return series.groupby(segment, group_keys=False).transform(
        lambda x: x.ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()
    )


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _wma(series: pd.Series, window: int) -> pd.Series:
    idx = pd.Series(np.arange(len(series), dtype=np.float64), index=series.index)
    rolling_sum = series.rolling(window, min_periods=window).sum()
    rolling_ix = (series * idx).rolling(window, min_periods=window).sum()
    return (
        rolling_ix - (idx - window) * rolling_sum
    ) / (window * (window + 1) / 2)


def _hma(series: pd.Series, window: int) -> pd.Series:
    half = max(2, window // 2)
    root = max(2, int(round(math.sqrt(window))))
    return _wma(
        2 * _wma(series, half) - _wma(series, window),
        root,
    )


def _rsi(return_series: pd.Series, window: int) -> pd.Series:
    up = return_series.clip(lower=0)
    down = -return_series.clip(upper=0)

    segment = return_series.isna().cumsum()

    avg_up = up.groupby(segment, group_keys=False).transform(
        lambda x: x.ewm(
            alpha=1 / window,
            adjust=False,
            min_periods=window,
        ).mean()
    )
    avg_down = down.groupby(segment, group_keys=False).transform(
        lambda x: x.ewm(
            alpha=1 / window,
            adjust=False,
            min_periods=window,
        ).mean()
    )

    denom = avg_up + avg_down
    return 100.0 * avg_up / denom.where(denom > 0)


def prepare_minute_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce sorted unique 1-minute rows.

    This function does NOT forward-fill missing candles.
    Missing timestamps become NaN rows so features crossing a gap become invalid.
    """
    required = {"timestamp", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    out = df[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ].copy()

    out["timestamp"] = out["timestamp"].astype("int64")
    out = (
        out.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    start = int(out["timestamp"].iloc[0])
    end = int(out["timestamp"].iloc[-1])

    if start % 60 != 0 or end % 60 != 0:
        raise ValueError(
            "Expected minute-aligned Unix timestamps in seconds"
        )

    full_timestamp = np.arange(start, end + 60, 60, dtype=np.int64)
    out = (
        out.set_index("timestamp")
        .reindex(full_timestamp)
        .rename_axis("timestamp")
        .reset_index()
    )

    return out


def build_ohlcv_features(df: pd.DataFrame) -> FeatureData:
    """
    Build all features directly from OHLCV.

    No VWAP or trade-level data is used.
    """
    base = prepare_minute_ohlcv(df)
    idx = base.index

    o = pd.Series(base["Open"].to_numpy(float), index=idx)
    h = pd.Series(base["High"].to_numpy(float), index=idx)
    l = pd.Series(base["Low"].to_numpy(float), index=idx)
    c = pd.Series(base["Close"].to_numpy(float), index=idx)
    v = pd.Series(base["Volume"].to_numpy(float), index=idx)

    observed = (
        np.isfinite(base[["Open", "High", "Low", "Close", "Volume"]]).all(axis=1)
        & (o > 0)
        & (h > 0)
        & (l > 0)
        & (c > 0)
        & (v >= 0)
        & (h >= np.maximum.reduce((o, c, l)))
        & (l <= np.minimum.reduce((o, c, h)))
    ).to_numpy(bool)

    invalid = ~observed
    for s in (o, h, l, c, v):
        s.loc[invalid] = np.nan

    log_close = np.log(c)
    return1 = log_close.diff()
    return5 = log_close.diff(5)

    candle_return = np.log(c / o)
    log_range = np.log(h / l)

    abs_range = h - l
    body_over_range = (c - o) / abs_range.where(abs_range > 0)
    close_position = (c - l) / abs_range.where(abs_range > 0) - 0.5
    upper_wick = (h - np.maximum(o, c)) / abs_range.where(abs_range > 0)
    lower_wick = (np.minimum(o, c) - l) / abs_range.where(abs_range > 0)

    log_volume = np.log1p(v)
    log_volume_delta = log_volume.diff()

    volume_mean60 = _rolling_mean(log_volume, 60)
    volume_std60 = _rolling_std(log_volume, 60)
    relative_log_volume60 = log_volume - volume_mean60
    volume_z60 = (
        relative_log_volume60
        / volume_std60.where(volume_std60 > 1e-8)
    )

    rv5 = np.sqrt(_rolling_mean(return1.pow(2), 5))
    rv60 = np.sqrt(_rolling_mean(return1.pow(2), 60))
    log_rv5_rv60 = np.log(rv5 / rv60.where(rv60 > 0))

    rsi15 = _rsi(return1, 15) / 100.0 - 0.5

    ema5 = _ema(log_close, 5)
    ema20 = _ema(log_close, 20)
    macd = ema5 - ema20
    signal7 = _ema(macd, 7)
    macd_hist = macd - signal7
    macd_hist_volnorm = macd_hist / rv60.where(rv60 > 0)
    macd_hist_acceleration = macd_hist_volnorm.diff().diff()

    ema8 = _ema(log_close, 8)
    ema32 = _ema(log_close, 32)
    ema_gap = (ema8 - ema32) / rv60.where(rv60 > 0)

    hma16 = _hma(log_close, 16)
    hma_slope16 = hma16.diff() / rv60.where(rv60 > 0)

    sign = np.sign(return1)
    sign_flip = (
        (sign != sign.shift(1))
        .where(return1.notna() & return1.shift(1).notna())
    )
    sign_flip_rate8 = _rolling_mean(sign_flip.astype(float), 8)

    minute = (base["timestamp"].to_numpy(np.int64) // 60) % 1440
    minute_mod5 = minute % 5

    minute_mod5_sin = pd.Series(
        np.sin(2 * np.pi * minute_mod5 / 5),
        index=idx,
    )
    minute_mod5_cos = pd.Series(
        np.cos(2 * np.pi * minute_mod5 / 5),
        index=idx,
    )
    time_of_day_sin = pd.Series(
        np.sin(2 * np.pi * minute / 1440),
        index=idx,
    )
    time_of_day_cos = pd.Series(
        np.cos(2 * np.pi * minute / 1440),
        index=idx,
    )

    fine_columns = (
        return1,
        return5,
        candle_return,
        log_range,
        body_over_range,
        close_position,
        upper_wick,
        lower_wick,
        log_volume_delta,
        relative_log_volume60,
        volume_z60,
        log_rv5_rv60,
        minute_mod5_sin,
        minute_mod5_cos,
        time_of_day_sin,
        time_of_day_cos,
        rsi15,
        macd_hist_volnorm,
        macd_hist_acceleration,
        ema_gap,
        hma_slope16,
        sign_flip_rate8,
    )

    fine = np.column_stack(
        [x.to_numpy(np.float32) for x in fine_columns]
    ).astype(np.float32)

    # ---------------- coarse 8-minute rolling state ----------------
    open8 = o.shift(7)
    high8 = h.rolling(8, min_periods=8).max()
    low8 = l.rolling(8, min_periods=8).min()
    volume8 = v.rolling(8, min_periods=8).sum()

    range8_abs = high8 - low8

    return8 = log_close.diff(8)
    return32 = log_close.diff(32)
    body8 = np.log(c / open8)
    range8 = np.log(high8 / low8)
    close_position8 = (
        (c - low8) / range8_abs.where(range8_abs > 0) - 0.5
    )

    log_volume8 = np.log1p(volume8)
    log_volume8_delta = log_volume8.diff(8)

    volume8_mean64 = _rolling_mean(log_volume8, 64)
    relative_log_volume8_64 = log_volume8 - volume8_mean64

    rv8 = np.sqrt(_rolling_mean(return1.pow(2), 8))
    rv64 = np.sqrt(_rolling_mean(return1.pow(2), 64))
    log_rv8_rv64 = np.log(rv8 / rv64.where(rv64 > 0))

    rsi64 = _rsi(return1, 64) / 100.0 - 0.5

    ema16 = _ema(log_close, 16)
    ema64 = _ema(log_close, 64)
    macd_long = ema16 - ema64
    signal16 = _ema(macd_long, 16)
    macd_hist_long = (macd_long - signal16) / rv64.where(rv64 > 0)

    ema32_long = _ema(log_close, 32)
    ema128 = _ema(log_close, 128)
    ema_gap_long = (
        (ema32_long - ema128) / rv64.where(rv64 > 0)
    )

    sign_flip_rate32 = _rolling_mean(
        sign_flip.astype(float),
        32,
    )

    coarse_columns = (
        return8,
        return32,
        body8,
        range8,
        close_position8,
        log_volume8_delta,
        relative_log_volume8_64,
        rv8,
        rv64,
        log_rv8_rv64,
        time_of_day_sin,
        time_of_day_cos,
        rsi64,
        macd_hist_long,
        ema_gap_long,
        sign_flip_rate32,
    )

    coarse = np.column_stack(
        [x.to_numpy(np.float32) for x in coarse_columns]
    ).astype(np.float32)

    # ---------------- targets ----------------
    close_np = c.to_numpy(np.float64)
    target = np.full(
        (len(base), len(HORIZONS)),
        np.nan,
        dtype=np.float32,
    )

    for j, horizon in enumerate(HORIZONS):
        valid = np.arange(len(base) - horizon)
        target[valid, j] = np.log(
            close_np[valid + horizon] / close_np[valid]
        ).astype(np.float32)

    rv60_np = rv60.to_numpy(np.float32)

    # Need enough history for the furthest lag and finite features/targets.
    min_lag = min(min(FINE_LAGS), min(COARSE_LAGS))
    eligible = np.ones(len(base), dtype=bool)
    eligible[: -min_lag] = False

    origins = np.flatnonzero(eligible)

    fine_ok = np.logical_and.reduce([
        np.isfinite(fine[origins + lag]).all(axis=1)
        for lag in FINE_LAGS
    ])
    coarse_ok = np.logical_and.reduce([
        np.isfinite(coarse[origins + lag]).all(axis=1)
        for lag in COARSE_LAGS
    ])

    eligible[origins] &= (
        fine_ok
        & coarse_ok
        & np.isfinite(rv60_np[origins])
        & np.isfinite(target[origins]).all(axis=1)
        & observed[origins]
    )

    return FeatureData(
        frame=base,
        fine=fine,
        coarse=coarse,
        rv60=rv60_np,
        target=target,
        eligible=eligible,
    )


def lgbm_feature_names() -> tuple[str, ...]:
    names = []

    for lag in FINE_LAGS:
        label = "t" if lag == 0 else f"t{lag:+d}m"
        names.extend(
            f"fine:{label}:{name}"
            for name in FINE_FEATURE_NAMES
        )

    for lag in COARSE_LAGS:
        label = "t" if lag == 0 else f"t{lag:+d}m"
        names.extend(
            f"coarse:{label}:{name}"
            for name in COARSE_FEATURE_NAMES
        )

    names.extend(("origin:rv60", "origin:log_rv60"))
    return tuple(names)


def build_lgbm_matrix(
    data: FeatureData,
    idx: np.ndarray,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)

    if not data.eligible[idx].all():
        bad = int((~data.eligible[idx]).sum())
        raise ValueError(
            f"{bad} selected rows are not strictly eligible"
        )

    parts = []

    for lag in FINE_LAGS:
        parts.append(data.fine[idx + lag])

    for lag in COARSE_LAGS:
        parts.append(data.coarse[idx + lag])

    rv = data.rv60[idx][:, None]
    parts.append(rv)
    parts.append(
        np.log(np.maximum(rv, 1e-12)).astype(np.float32)
    )

    x = np.concatenate(parts, axis=1).astype(np.float32)

    if x.shape[1] != len(lgbm_feature_names()):
        raise AssertionError("Feature-width mismatch")

    if not np.isfinite(x).all():
        raise ValueError("Non-finite LGBM matrix")

    return x


def assert_p100_lightgbm(config: LGBMConfig) -> None:
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if smi.returncode != 0 or not smi.stdout.strip():
        raise RuntimeError("No NVIDIA GPU detected")

    if config.require_p100 and "P100" not in smi.stdout.upper():
        raise RuntimeError(
            f"P100 required, detected: {smi.stdout.strip()}"
        )

    rng = np.random.default_rng(config.seed)
    x = rng.normal(size=(2048, 16)).astype(np.float32)
    y = (
        0.4 * x[:, 0]
        - 0.2 * x[:, 1]
        + rng.normal(scale=0.1, size=len(x))
    ).astype(np.float32)

    probe = lgb.LGBMRegressor(
        objective="huber",
        alpha=config.alpha,
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=15,
        max_bin=config.max_bin,
        device_type=config.device_type,
        gpu_use_dp=config.gpu_use_dp,
        verbosity=-1,
        random_state=config.seed,
    )

    try:
        probe.fit(
            x,
            y,
            callbacks=[lgb.log_evaluation(period=0)],
        )
    except Exception as exc:
        raise RuntimeError(
            "LightGBM GPU preflight failed; CPU fallback disabled"
        ) from exc


def _make_model(
    config: LGBMConfig,
    seed: int,
    n_estimators: int | None = None,
):
    return lgb.LGBMRegressor(
        objective=config.objective,
        alpha=config.alpha,
        n_estimators=int(
            n_estimators
            if n_estimators is not None
            else config.n_estimators
        ),
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        max_bin=config.max_bin,
        device_type=config.device_type,
        gpu_use_dp=config.gpu_use_dp,
        verbosity=-1,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        n_jobs=config.n_jobs,
    )


def fit_lgbm_baseline(
    data: FeatureData,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    early_idx: np.ndarray | None = None,
    config: LGBMConfig | None = None,
    fixed_rounds: int | Sequence[int] | None = None,
) -> LGBMResult:
    """
    Fit the OHLCV-only baseline.

    train_idx / early_idx / predict_idx are integer row indices into data.frame.
    """
    config = config or LGBMConfig()

    train_idx = np.asarray(train_idx, dtype=np.int64)
    predict_idx = np.asarray(predict_idx, dtype=np.int64)
    if early_idx is not None:
        early_idx = np.asarray(early_idx, dtype=np.int64)

    transform = TargetTransform.fit(
        data.target,
        data.rv60,
        train_idx,
    )

    x_train = build_lgbm_matrix(data, train_idx)
    y_train = transform.encode(
        data.target,
        data.rv60,
        train_idx,
    )

    x_pred = build_lgbm_matrix(data, predict_idx)

    x_early = None
    y_early = None
    if early_idx is not None:
        x_early = build_lgbm_matrix(data, early_idx)
        y_early = transform.encode(
            data.target,
            data.rv60,
            early_idx,
        )

    if fixed_rounds is None:
        rounds = [None] * len(HORIZONS)
    elif np.isscalar(fixed_rounds):
        rounds = [int(fixed_rounds)] * len(HORIZONS)
    else:
        rounds = [int(x) for x in fixed_rounds]
        if len(rounds) != len(HORIZONS):
            raise ValueError("fixed_rounds must have length 3")

    models = []
    best_iterations = []
    z_pred = np.empty(
        (len(predict_idx), len(HORIZONS)),
        dtype=np.float32,
    )

    for col, horizon in enumerate(HORIZONS):
        model = _make_model(
            config,
            seed=config.seed + 101 * col,
            n_estimators=rounds[col],
        )

        fit_kwargs = {
            "callbacks": [lgb.log_evaluation(period=0)]
        }

        if rounds[col] is None and x_early is not None:
            fit_kwargs["eval_set"] = [
                (x_early, y_early[:, col])
            ]
            fit_kwargs["eval_metric"] = "huber"
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(
                    config.early_stopping_rounds,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ]

        model.fit(
            x_train,
            y_train[:, col],
            **fit_kwargs,
        )

        best_iter = int(
            getattr(model, "best_iteration_", 0)
            or rounds[col]
            or config.n_estimators
        )

        z_pred[:, col] = model.predict(
            x_pred,
            num_iteration=best_iter,
        ).astype(np.float32)

        models.append(model)
        best_iterations.append(best_iter)

    prediction = transform.decode(
        z_pred,
        data.rv60[predict_idx],
    )

    names = lgbm_feature_names()
    importance_rows = []

    for horizon, model in zip(HORIZONS, models):
        gain = model.booster_.feature_importance(
            importance_type="gain"
        )
        split = model.booster_.feature_importance(
            importance_type="split"
        )

        total_gain = float(np.sum(gain))

        for name, g, s in zip(names, gain, split):
            importance_rows.append({
                "horizon": horizon,
                "feature": name,
                "gain": float(g),
                "gain_share_pct": (
                    100.0 * float(g) / total_gain
                    if total_gain > 0
                    else 0.0
                ),
                "split_count": int(s),
            })

    return LGBMResult(
        models=models,
        prediction=prediction,
        encoded_prediction=z_pred,
        feature_names=names,
        importance=pd.DataFrame(importance_rows),
        best_iterations=tuple(best_iterations),
        target_transform=transform,
    )


def timestamp_indices(
    data: FeatureData,
    start_timestamp: int,
    end_exclusive_timestamp: int,
) -> np.ndarray:
    ts = data.frame["timestamp"].to_numpy(np.int64)
    return np.flatnonzero(
        data.eligible
        & (ts >= int(start_timestamp))
        & (ts < int(end_exclusive_timestamp))
    ).astype(np.int32)


if __name__ == "__main__":
    print("OHLCV-only LightGBM baseline")
    print(f"features = {len(lgbm_feature_names())}")
    print(f"fine base features = {len(FINE_FEATURE_NAMES)}")
    print(f"coarse base features = {len(COARSE_FEATURE_NAMES)}")
    print("VWAP required = False")
