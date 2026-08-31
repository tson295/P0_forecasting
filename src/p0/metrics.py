"""Metric §0: model dự báo log-return ŷ_h; chấm trên GIÁ P̂ = C_t·exp(ŷ_h).

- RMSE_h, MAE_h trên e_h = P̂_{t+h} − C_{t+h} (USD)
- Pearson r và directional accuracy trên thay đổi giá (P̂ − C_t) vs (C_{t+h} − C_t); dir-acc bỏ bar C_{t+h} = C_t
- Gain = 1 − RMSE_cand / RMSE_base (pp); tóm tắt MedianGain / WinRate / P10Gain / WorstGain trên 15 ô
- Gộp 3 seed (§2.1b): mỗi ô lấy MEAN RMSE của các seed → bảng RMSE̅; Gain tính từ RMSE̅; MedianGain = median 15 ô
"""
from __future__ import annotations

import numpy as np

from .config import HORIZONS


def price_from_logret(c_t: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """P̂_{t+h} = C_t · exp(ŷ_h); c_t (n,), yhat (n, 3) → (n, 3)."""
    return c_t[:, None] * np.exp(yhat)


def cell_metrics(c_t: np.ndarray, c_future: np.ndarray, yhat: np.ndarray) -> dict[str, np.ndarray]:
    """Metric per horizon cho một tập origin. c_future (n,3) = C_{t+h}; yhat (n,3) log-return dự báo."""
    p_hat = price_from_logret(c_t, yhat)
    err = p_hat - c_future
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    true_chg = c_future - c_t[:, None]
    pred_chg = p_hat - c_t[:, None]
    r = np.zeros(len(HORIZONS))
    dacc = np.zeros(len(HORIZONS))
    for j in range(len(HORIZONS)):
        a, b = pred_chg[:, j], true_chg[:, j]
        r[j] = float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else 0.0
        nz = b != 0
        dacc[j] = float(np.mean(np.sign(a[nz]) == np.sign(b[nz]))) if nz.any() else np.nan
    return {"rmse": rmse, "mae": mae, "r": r, "dir_acc": dacc}


def e0_rmse(c_t: np.ndarray, c_future: np.ndarray) -> np.ndarray:
    """E0: P̂ = C_t → RMSE_h = sqrt(mean((C_{t+h} − C_t)²))."""
    return np.sqrt(np.mean((c_future - c_t[:, None]) ** 2, axis=0))


def gain_pp(rmse_cand: np.ndarray, rmse_base: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 - np.asarray(rmse_cand, float) / np.asarray(rmse_base, float))


def summarize(gains: np.ndarray) -> dict[str, float]:
    g = np.asarray(gains, float).ravel()
    return {
        "MedianGain": float(np.median(g)),
        "WinRate": float(np.mean(g > 0)),
        "P10Gain": float(np.percentile(g, 10)),
        "WorstGain": float(np.min(g)),
        "n_cells": int(g.size),
    }


def mean_rmse_over_seeds(tables: list[np.ndarray]) -> np.ndarray:
    """Bảng RMSE̅ 15 ô: mỗi ô = mean RMSE của các seed (§2.1b)."""
    arr = np.stack([np.asarray(t, float) for t in tables])
    return arr.mean(axis=0)


def decide(median_gain: float, eps: float) -> str:
    """Luật §2.1: MedianGain ≥ −ε → KEEP (tốt hơn hoặc gần như không đổi); < −ε → DROP."""
    return "KEEP" if median_gain >= -eps else "DROP"


def seed_noise_cells(rmse_tables: list[np.ndarray]) -> np.ndarray:
    """Nhiễu seed từng ô (§1.3), đơn vị pp: với S evaluation seed cho ô (fold, horizon) có RMSE R_1..R_S,
    mu = mean(R), sigma = std(R, ddof=0) → noise_cell = 100·sigma/mu. KHÔNG seed nào làm mốc/mẫu số."""
    arr = np.stack([np.asarray(t, float) for t in rmse_tables])  # (S, F, 3)
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0, ddof=0)
    return 100.0 * sigma / mu


def seed_noise_eps(rmse_tables: list[np.ndarray], floor_pp: float = 0.005) -> float:
    """ε = max(floor, sqrt(mean(noise_cell²))) — RMS của nhiễu seed trên 15 ô (cùng đơn vị pp với Gain)."""
    if len(rmse_tables) < 2:
        return float(floor_pp)
    return float(max(floor_pp, np.sqrt(np.mean(seed_noise_cells(rmse_tables) ** 2))))
