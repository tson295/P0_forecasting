"""Dữ liệu 1 phút tổng hợp (random walk) cho unit/smoke test CPU — KHÔNG phải data thật, không dùng cho kết quả."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_hf(n_days: float = 4.0, start: str = "2026-01-01 00:00:00", seed: int = 0, price0: float = 80_000.0, sig: float = 7.65e-4,
            ar1: float = -0.06) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(n_days * 1440)
    eps = rng.normal(0.0, sig, n)
    r = np.empty(n)
    r[0] = eps[0]
    for i in range(1, n):  # AR(1) âm nhẹ như BTC 1 phút → có chút tín hiệu cho model học
        r[i] = ar1 * r[i - 1] + eps[i]
    close = price0 * np.exp(np.cumsum(r))
    open_ = np.r_[price0, close[:-1]]
    wick = np.abs(rng.normal(0.0, 0.4 * sig, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.lognormal(0.0, 0.6, n)
    vwap = (high + low + close) / 3.0 + rng.uniform(-0.2, 0.2, n) * (high - low)
    amount = vwap * volume
    ts0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    ts = ts0 + 60 * np.arange(n)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "amount": amount})
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df[["datetime", "timestamp", "open", "high", "low", "close", "volume", "amount"]]


def make_lf(hf: pd.DataFrame) -> pd.DataFrame:
    """Gộp 5 bar 1 phút thành bar 5' với nhãn T = bar cuối (T−4..T], đúng convention của data thật."""
    g = hf.copy()
    # nhóm sao cho bar cuối của nhóm có nhãn bội số 300 s (16:16..16:20 → T = 16:20), như data thật
    g["grp"] = ((g["timestamp"] // 60) - 1) // 5
    g = g[g.groupby("grp")["timestamp"].transform("size") == 5]  # bỏ nhóm 5' chưa đóng ở đầu/cuối chuỗi
    agg = g.groupby("grp").agg(timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"), low=("low", "min"),
                               close=("close", "last"), volume=("volume", "sum"), amount=("amount", "sum")).reset_index(drop=True)
    agg["datetime"] = pd.to_datetime(agg["timestamp"], unit="s", utc=True)
    return agg[["datetime", "timestamp", "open", "high", "low", "close", "volume", "amount"]]
