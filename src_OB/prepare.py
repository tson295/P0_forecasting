"""Read fixed historical L2 files in chunks; no real-time stream at training."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json, START, END, START_US, END_US

DTYPES = {"raw_ts": "int64", "raw_mid": "float64", "raw_segment": "int64",
          "ts": "int64", "mid": "float64", "segment": "int64", "features": "float32"}
def feature_names(include_distances=False):
    fields = ["of_bid", "of_ask", "ofi"]
    if include_distances:
        fields += ["bid_distance_bps", "ask_distance_bps"]
    return ([f"{field}_{level}" for level in range(10) for field in fields]
            + ["log_elapsed", "log_raw_elapsed", "log_updates"])


def prepare(cfg):
    dest = Path(cfg["prepared_dir"])
    first, last = date.fromisoformat(START), date.fromisoformat(END)
    files = [Path(cfg["raw_dir"]) / f"{first + timedelta(days=i)}.csv.gz"
             for i in range((last - first).days)]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Historical dataset phải đủ 730 ngày; thiếu {len(missing)} ngày, đầu tiên: {missing[0]}")
    dest.mkdir(parents=True, exist_ok=False)
    cols = ["exchange", "symbol", "timestamp", "local_timestamp"] + [
        f"{side}[{i}].{field}" for side in ("bids", "asks") for i in range(10)
        for field in ("price", "amount")]
    previous = None
    last_kept_ts = None
    segment = -1
    totals = {"raw": 0, "kept": 0}
    pending = np.zeros((10, 2), np.float64)
    pending_count = 0
    with ExitStack() as stack:
        handles = {key: stack.enter_context((dest / f"{key}.bin").open("wb")) for key in DTYPES}
        for path in files:
            for frame in pd.read_csv(path, usecols=cols, chunksize=cfg["chunk_rows"]):
                if not (frame.symbol.eq(cfg["symbol"]).all() and frame.exchange.eq(cfg["exchange"]).all()):
                    raise ValueError("Dữ liệu không khớp symbol/exchange trong config.")
                ts = frame.local_timestamp.to_numpy(np.int64)
                bp, bq, ap, aq = [frame[[f"{side}[{i}].{field}" for i in range(10)]].to_numpy(float)
                                   for side, field in (("bids", "price"), ("bids", "amount"),
                                                       ("asks", "price"), ("asks", "amount"))]
                raw_rows, kept_rows, features = [], [], []
                for i in range(len(ts)):
                    now = int(ts[i])
                    if not START_US <= now < END_US:
                        continue
                    values = np.stack((bp[i], bq[i], ap[i], aq[i]))
                    if (not np.isfinite(values).all() or (values[[0, 2]] <= 0).any()
                            or (values[[1, 3]] < 0).any() or bp[i, 0] >= ap[i, 0]
                            or (np.diff(bp[i]) >= 0).any() or (np.diff(ap[i]) <= 0).any()):
                        previous = None  # next usable row starts a new segment
                        continue
                    if previous is not None and now < previous[0]:
                        raise ValueError("local_timestamp đảo thứ tự; không sort để che lỗi feed.")
                    restart = previous is None or now - previous[0] > cfg["max_feed_gap_seconds"] * 1e6
                    mid = (bp[i, 0] + ap[i, 0]) / 2
                    if restart:
                        segment += 1
                        pending.fill(0)
                        pending_count = 0
                        last_kept_ts = now
                        raw_dt = 0.
                        keep = True
                    else:
                        old_ts, old_bp, old_bq, old_ap, old_aq, old_mid = previous
                        # Ask flow is signed liquidity supply; OFI = bid flow - ask flow.
                        bf = np.where(bp[i] > old_bp, bq[i],
                                      np.where(bp[i] == old_bp, bq[i] - old_bq, -old_bq))
                        af = np.where(ap[i] < old_ap, aq[i],
                                      np.where(ap[i] == old_ap, aq[i] - old_aq, -old_aq))
                        # Accumulate liquidity units before any normalization or deduplication.
                        pending += np.stack((bf, af), axis=1)
                        raw_dt = (now - old_ts) / 1e6
                        keep = mid != old_mid
                    pending_count += 1
                    raw_rows.append((now, mid, segment))
                    if keep:
                        columns = [pending, pending[:, 0] - pending[:, 1]]
                        if cfg.get("include_distances", False):
                            columns += [1e4 * (bp[i] / mid - 1), 1e4 * (ap[i] / mid - 1)]
                        level_features = np.column_stack(columns).reshape(-1)
                        time_features = np.log1p([(now - last_kept_ts) / 1e6, raw_dt, pending_count])
                        features.append(np.concatenate((level_features, time_features)))
                        kept_rows.append((now, mid, segment))
                        last_kept_ts = now
                        pending.fill(0)
                        pending_count = 0
                    previous = (now, bp[i].copy(), bq[i].copy(), ap[i].copy(), aq[i].copy(), mid)
                for rows, prefix in ((raw_rows, "raw_"), (kept_rows, "")):
                    if rows:
                        for j, key in enumerate((prefix + "ts", prefix + "mid", prefix + "segment")):
                            np.asarray([r[j] for r in rows], dtype=DTYPES[key]).tofile(handles[key])
                if features:
                    np.asarray(features, np.float32).tofile(handles["features"])
                totals["raw"] += len(raw_rows)
                totals["kept"] += len(kept_rows)
            print(f"prepared {path.name}: {totals}", flush=True)
    write_json(dest / "manifest.json", {"config": cfg, "schema_version": 2,
               "start_inclusive": START, "end_exclusive": END, "historical_fixed": True,
               "counts": totals, "features": feature_names(cfg.get("include_distances", False)),
               "dtypes": DTYPES, "files": [p.name for p in files], "timestamp_unit": "microseconds",
               "price": "L2 mid", "of": "per-level price-aware bid/ask flow before mid dedup",
               "source": "vendor reconstructed book_snapshot_25, first 10 levels",
               "sequence_ids_available": False})
