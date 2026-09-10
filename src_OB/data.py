from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400 * 1_000_000


@dataclass
class Fold:
    name: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int


class Data:
    def __init__(self, cfg, require_current_replay=True):
        from .reconstruct import REPLAY_VERSION

        self.cfg = cfg
        folder = Path(cfg["prepared_dir"])
        self.meta = json.loads((folder / "manifest.json").read_text())
        if (self.meta.get("schema_version") != 3 or self.meta.get("dataset_repo") != cfg["dataset_repo"]
                or self.meta.get("dataset_revision") != cfg["dataset_revision"]):
            raise ValueError("Cần reconstructed HF dataset schema v3 đúng revision.")
        # Training must not silently use data built by an older replay (checker R6-W1); reports may read it.
        if require_current_replay and self.meta.get("replay_version") != REPLAY_VERSION:
            raise ValueError(f"Prepared data dùng replay {self.meta.get('replay_version')}, cần {REPLAY_VERSION}; "
                             "prepare vào prepared_dir mới trước khi train.")
        segments = json.loads((folder / "segments.json").read_text())
        self.segment_start = np.asarray([s["start_us"] for s in segments], np.int64)
        self.segment_end = np.asarray([s["end_exclusive_us"] for s in segments], np.int64)
        if self.meta["config"].get("include_distances", False) != cfg.get("include_distances", False):
            raise ValueError("Feature distance config khác prepared dataset; tạo prepared_dir riêng.")
        if any(self.meta["config"][k] != cfg[k] for k in
               ("symbol", "exchange", "levels", "book_max_depth", "max_feed_gap_seconds")):
            raise ValueError("Prepared data không khớp thị trường/độ sâu/segment policy trong config.")
        self.width = len(self.meta["features"])
        for key, dtype in self.meta["dtypes"].items():
            shape = ((self.meta["counts"]["kept"], self.width) if key == "features" else
                     (self.meta["counts"]["raw" if key.startswith("raw_") else "kept"],))
            setattr(self, key, np.memmap(folder / f"{key}.bin", mode="r", dtype=dtype, shape=shape))

    def folds(self):
        cfg = self.cfg
        coverage = self.meta["coverage"]
        start, end = coverage["start_inclusive_us"], coverage["end_exclusive_us"]
        required = (cfg["train_days"] + cfg["gap_days"] + cfg["val_days"]) * DAY
        available = int((end - start - required) // (cfg["step_days"] * DAY)) + 1
        count = min(cfg["n_folds"], max(0, available))
        if not count:
            raise ValueError("Coverage thực tế không đủ train/gap/VAL; không pad lịch sử bị thiếu.")
        print(f"walk-forward: {count} folds from observed coverage, requested maximum={cfg['n_folds']}", flush=True)
        for n in range(count):
            val_end = end - (count - 1 - n) * cfg["step_days"] * DAY
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
        # Never extend a quote past the last accepted state of a closed segment,
        # even when the next snapshot is less than max_price_age_seconds away.
        valid &= queries < self.segment_end[self.raw_segment[clipped]]
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
        price, valid, raw_ids = self.prices_at(self.ts[ids] + horizon * 1_000_000)
        valid &= self.raw_segment[raw_ids] == self.segment[ids]
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
        prices, valid, raw_ids = self.prices_at(times)
        valid &= self.raw_segment[raw_ids] == self.segment[ids, None]
        return np.log(prices), valid.all(axis=1)

    def covariates_at(self, queries):
        ids = np.searchsorted(self.ts, queries, side="right") - 1
        clipped = np.maximum(ids, 0)
        _, valid, raw_ids = self.prices_at(queries)
        valid &= (ids >= 0) & (self.segment[clipped] == self.raw_segment[raw_ids])
        return np.asarray(self.features[clipped], np.float64), valid


def select_origins(cfg, data, fold, counts=None):
    """Common FIT/VAL origin population for every family; train and DATA_REPORT share it.

    Label and context stay in one valid segment. When TimesFM/AutoTS are in the configured
    model list, their h-spaced price context must be intact too; --models never changes this.
    `counts` receives the number of origins left after each mask (cumulative over horizons).
    """
    counts = {} if counts is None else counts
    train_ids = data.indices(fold.train_start, fold.train_end)
    counts["train_label_context"] = len(train_ids)
    train_ids = train_ids[data.ts[train_ids - cfg["context"] + 1] >= fold.train_start]
    counts["train_context_inside_fit"] = len(train_ids)
    val_ids = data.indices(fold.val_start, fold.val_end)
    counts["val_label_context"] = len(val_ids)
    population_models = cfg["models"]  # --models splits jobs without changing their scoring population
    if any(m.startswith("tfm") or m == "autots" for m in population_models):
        length = max(cfg["tfm"]["context"] if any(m.startswith("tfm") for m in population_models) else 1,
                     cfg["autots"]["max_window_size"] if "autots" in population_models else 1)
        counts["price_context_points"] = length
        for horizon in cfg["horizons_seconds"]:
            def eligible(ids):
                keep = []
                for s in range(0, len(ids), 4096):
                    batch = ids[s:s + 4096]
                    _, valid = data.price_context(batch, horizon, length)
                    keep.append(batch[valid])
                return np.concatenate(keep) if keep else np.empty(0, np.int64)
            train_ids, val_ids = eligible(train_ids), eligible(val_ids)
            train_ids = train_ids[data.ts[train_ids] - (length - 1) * horizon * 1_000_000 >= fold.train_start]
            counts[f"train_price_context_h{horizon}s"] = len(train_ids)
            counts[f"val_price_context_h{horizon}s"] = len(val_ids)
    return train_ids, val_ids


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
