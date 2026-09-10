from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from .config import START, END, END_US

DAY = 86400 * 1_000_000


@dataclass
class Fold:
    name: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int


class Data:
    def __init__(self, cfg):
        self.cfg = cfg
        folder = Path(cfg["prepared_dir"])
        self.meta = json.loads((folder / "manifest.json").read_text())
        if (self.meta.get("schema_version") != 2 or
                (self.meta.get("start_inclusive"), self.meta.get("end_exclusive")) != (START, END)):
            raise ValueError("Cần prepared dataset OF/OFI schema v2 trên đúng 2 năm đã freeze.")
        if self.meta["config"].get("include_distances", False) != cfg.get("include_distances", False):
            raise ValueError("Feature distance config khác prepared dataset; tạo prepared_dir riêng.")
        if any(self.meta["config"][k] != cfg[k] for k in ("symbol", "exchange", "levels")):
            raise ValueError("Prepared data không khớp thị trường/độ sâu trong config.")
        self.width = len(self.meta["features"])
        for key, dtype in self.meta["dtypes"].items():
            shape = ((self.meta["counts"]["kept"], self.width) if key == "features" else
                     (self.meta["counts"]["raw" if key.startswith("raw_") else "kept"],))
            setattr(self, key, np.memmap(folder / f"{key}.bin", mode="r", dtype=dtype, shape=shape))

    def folds(self):
        cfg = self.cfg
        end = END_US
        for n in range(cfg["n_folds"]):
            val_end = end - (cfg["n_folds"] - 1 - n) * cfg["step_days"] * DAY
            val_start = val_end - cfg["val_days"] * DAY
            train_end = val_start - cfg["gap_days"] * DAY
            train_start = train_end - cfg["train_days"] * DAY
            if train_start < int(self.raw_ts[0]):
                raise ValueError("Lịch sử không đủ cho fold; điều chỉnh train_days/n_folds sau khi có dữ liệu.")
            yield Fold(f"fold{n + 1}", train_start, train_end, val_start, val_end)

    def prices_at(self, queries):
        queries = np.asarray(queries, dtype=np.int64)
        pos = np.searchsorted(self.raw_ts, queries, side="right") - 1
        clipped = np.maximum(pos, 0)
        valid = (pos >= 0) & (queries <= self.raw_ts[-1])
        valid &= queries - self.raw_ts[clipped] <= self.cfg["max_price_age_seconds"] * 1e6
        return np.asarray(self.raw_mid[clipped]), valid, clipped

    def indices(self, start, end):
        lo, hi = np.searchsorted(self.ts, [start, end], side="left")
        context = self.cfg["context"]
        chunks = []
        max_h = max(self.cfg["horizons_seconds"]) * 1_000_000
        for s in range(max(int(lo), context - 1), int(hi), 100000):
            ids = np.arange(s, min(s + 100000, hi))
            mask = self.ts[ids] + max_h < end
            mask &= self.segment[ids] == self.segment[ids - context + 1]
            # Resolve equal timestamp updates at the timestamp's final observed book.
            current, current_ok, raw_current = self.prices_at(self.ts[ids])
            mask &= current_ok & (current == self.mid[ids])
            mask &= self.raw_segment[raw_current] == self.segment[ids]
            for h in self.cfg["horizons_seconds"]:
                _, ok, raw_id = self.prices_at(self.ts[ids] + h * 1_000_000)
                mask &= ok & (self.raw_segment[raw_id] == self.segment[ids])
            chunks.append(ids[mask])
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def target(self, ids, horizon):
        price, valid, _ = self.prices_at(self.ts[ids] + horizon * 1_000_000)
        if not valid.all():
            raise ValueError("Nhãn không có quote hợp lệ tại thời điểm horizon.")
        return np.log(price / self.mid[ids]), price

    def windows(self, ids):
        offsets = np.arange(1 - self.cfg["context"], 1)
        return np.asarray(self.features[np.asarray(ids)[:, None] + offsets], np.float32)

    def flat(self, ids):
        # Fixed features only: latest state + short/long summaries + explicit OFI.
        x = self.windows(ids)
        ofi_columns = [self.meta["features"].index(f"ofi_{i}") for i in range(10)]
        ofi = x[..., ofi_columns]
        return np.concatenate((x[:, -1], x[:, -10:].mean(1), x.mean(1), ofi.sum(1)), axis=1)

    def price_context(self, ids, horizon, length):
        times = self.ts[ids, None] - np.arange(length - 1, -1, -1)[None, :] * horizon * 1_000_000
        prices, valid, _ = self.prices_at(times)
        return np.log(prices), valid.all(axis=1)


def price_metrics(actual, predicted):
    """All three metrics are evaluated on raw L2 mid prices, never log returns."""
    a, p = np.asarray(actual, np.float64), np.asarray(predicted, np.float64)
    if len(a) == 0 or not (np.isfinite(a).all() and np.isfinite(p).all()):
        raise ValueError("Không thể tính metric trên prediction rỗng/NaN/inf.")
    error = p - a
    sse, sst = float(error @ error), float(((a - a.mean()) ** 2).sum())
    return {"RMSE": float(np.sqrt(sse / len(a))), "MAE": float(np.abs(error).mean()),
            "R2": 1 - sse / sst if sst > 0 else None, "n": len(a), "price_unit": "USDT"}


def date_string(ts):
    return pd.Timestamp(ts, unit="us", tz="UTC").isoformat()
