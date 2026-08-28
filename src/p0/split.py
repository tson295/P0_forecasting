"""Split §1.2: partition half-open [T_start, T_end); origin t thuộc partition chỉ khi t ≥ T_start và t + 3' < T_end.

15 ngày: expanding FIT từ origin eligible đầu tiên, ES = ngày trước VAL (00:00 → VAL_start − purge), VAL = 1 ngày UTC,
TEST = 2 ngày cuối (refit: FIT → ngày trước TEST, ES = ngày trước TEST − purge).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HMAX_SEC

DAY_SEC = 86_400


def utc_ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


@dataclass(frozen=True)
class Partition:
    start: int  # unix giây, inclusive
    end: int  # unix giây, exclusive

    def origins(self, ts: np.ndarray, eligible: np.ndarray, hmax_sec: int = HMAX_SEC) -> np.ndarray:
        """Chỉ số các origin hợp lệ: eligible (B0) và toàn bộ target nằm trong partition."""
        mask = eligible & (ts >= self.start) & (ts + hmax_sec < self.end)
        return np.flatnonzero(mask).astype(np.int64)

    def label(self) -> str:
        f = lambda x: pd.Timestamp(x, unit="s", tz="UTC").strftime("%m-%d %H:%M")  # noqa: E731
        return f"{f(self.start)} → {f(self.end)}"


@dataclass(frozen=True)
class Fold:
    name: str
    fit: Partition
    es: Partition
    val: Partition  # VAL, hoặc TEST ở final

    def counts(self, ts: np.ndarray, eligible: np.ndarray) -> dict[str, int]:
        return {k: int(len(getattr(self, k).origins(ts, eligible))) for k in ("fit", "es", "val")}


def make_folds(first_origin_ts: int, val_days: list[str], purge_minutes: int = 60, es_hours: int | None = None) -> list[Fold]:
    """Fold 15 ngày (§1.2). ES = [D−1 00:00, VAL_start − purge); FIT = [first_origin, D−1 00:00) expanding."""
    folds = []
    for i, day in enumerate(val_days, start=1):
        val_start = utc_ts(day)
        val = Partition(val_start, val_start + DAY_SEC)
        es_start = val_start - DAY_SEC
        es_end = val_start - purge_minutes * 60
        if es_hours is not None:
            es_end = min(es_end, es_start + es_hours * 3600)
        es = Partition(es_start, es_end)
        fit = Partition(first_origin_ts, es_start)
        folds.append(Fold(f"fold{i}_{day}", fit, es, val))
    return folds


def make_final(first_origin_ts: int, test_start: str, test_end_ts: int, purge_minutes: int = 60) -> Fold:
    """TEST (§4): refit FIT → ngày trước TEST 00:00; ES = ngày trước TEST − purge; TEST = [test_start, test_end)."""
    t0 = utc_ts(test_start)
    es_start = t0 - DAY_SEC
    es = Partition(es_start, t0 - purge_minutes * 60)
    fit = Partition(first_origin_ts, es_start)
    return Fold("final_TEST", fit, es, Partition(t0, test_end_ts))


def check_fold(fold: Fold, ts: np.ndarray, eligible: np.ndarray, purge_minutes: int = 60) -> dict:
    """Kiểm tra §6.3/§6.5: rời nhau, purge đủ, origin cuối = T_end − 4'."""
    f, e, v = fold.fit.origins(ts, eligible), fold.es.origins(ts, eligible), fold.val.origins(ts, eligible)
    problems = []
    if len(f) == 0 or len(e) == 0 or len(v) == 0:
        problems.append("partition rỗng")
    if len(set(f) & set(e)) or len(set(e) & set(v)) or len(set(f) & set(v)):
        problems.append("partition giao nhau")
    if len(f) and len(e) and ts[f].max() + HMAX_SEC >= fold.es.start:
        problems.append("target FIT lấn sang ES")
    if len(e) and len(v) and ts[e].max() + HMAX_SEC >= fold.es.end:
        problems.append("target ES lấn qua biên ES")
    if fold.val.start - fold.es.end < purge_minutes * 60:
        problems.append("purge < 60 phút")
    if len(v) and ts[v].max() + HMAX_SEC >= fold.val.end:
        problems.append("origin VAL cuối vi phạm t + 3' < T_end")
    last_is_tend_minus_4 = bool(len(v)) and int(ts[v].max()) == fold.val.end - HMAX_SEC - 60  # thông tin: bar T_end − 4' có eligible không
    return {"fold": fold.name, "n_fit": int(len(f)), "n_es": int(len(e)), "n_val": int(len(v)), "problems": problems, "ok": not problems,
            "last_val_origin_is_Tend_minus_4min": last_is_tend_minus_4}
