"""TargetTransform — tái hiện ĐÚNG công thức của Baseline_LGBM.TargetTransform (train-only, y / (rv60·√h), chuẩn hoá mean/scale).

Lý do tồn tại: bản trong Baseline_LGBM.py dùng phép nhân in-place `denom *= sqrt(H)[None, :]` với denom shape (n, 1) →
numpy không cho broadcast in-place sang (n, 3) → ValueError ở mọi phiên bản numpy; tức B0 gốc không chạy được nguyên bản.
File B0 bị đóng băng (không sửa), nên harness dùng bản này với phép nhân out-of-place; mọi giá trị (floor, mean, scale,
encode, decode) giữ nguyên định nghĩa của B0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HORIZONS

_SQRT_H = np.sqrt(np.asarray(HORIZONS, np.float32))[None, :]


@dataclass
class TargetTransform:
    mean: np.ndarray
    scale: np.ndarray
    volatility_floor: float

    @classmethod
    def fit(cls, y: np.ndarray, rv60: np.ndarray, train_idx: np.ndarray) -> "TargetTransform":
        train_rv = rv60[train_idx]
        floor = float(np.quantile(train_rv, 0.01))
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("Invalid RV60 floor")
        denom = np.maximum(train_rv, floor)[:, None] * _SQRT_H
        normalized = y[train_idx] / denom
        mean = np.mean(normalized, axis=0).astype(np.float32)
        scale = np.std(normalized, axis=0).astype(np.float32)
        scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
        return cls(mean, scale, floor)

    def _denom(self, rv: np.ndarray) -> np.ndarray:
        return np.maximum(rv, self.volatility_floor)[:, None] * _SQRT_H

    def encode(self, y: np.ndarray, rv60: np.ndarray, idx: np.ndarray) -> np.ndarray:
        normalized = y[idx] / self._denom(rv60[idx])
        return ((normalized - self.mean) / self.scale).astype(np.float32)

    def decode(self, z: np.ndarray, rv60: np.ndarray) -> np.ndarray:
        normalized = z * self.scale + self.mean
        return (normalized * self._denom(rv60)).astype(np.float32)
