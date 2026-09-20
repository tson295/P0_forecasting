"""EDA for a raw L10 book CSV: cadence, ordering, gaps, book validity, horizon targets.

Reports what the file IS, so the sampling contract can be derived from the measured
cadence instead of assumed. Writes JSON plus a short Markdown summary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

HORIZONS = (60, 120, 180)
LEVELS = range(1, 11)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def timestamp_column(columns):
    for name, unit in (("timestamp_ns", "ns"), ("timestamp_us", "us"),
                       ("timestamp_ms", "ms"), ("timestamp_s", "s")):
        if name in columns:
            return name, unit
    raise ValueError(f"No recognised timestamp column in {list(columns)[:8]}")


def ordering_report(ts):
    """File order vs chronological order, and the contiguous monotone blocks."""
    step = np.diff(ts)
    breaks = np.flatnonzero(step < 0)
    blocks = []
    for lo, hi in zip(np.r_[0, breaks+1], np.r_[breaks+1, len(ts)]):
        blocks.append(dict(start_row=int(lo), end_row=int(hi-1), rows=int(hi-lo),
                           first=int(ts[lo]), last=int(ts[hi-1])))
    return dict(file_order_monotonic=bool((step > 0).all()),
                backwards_steps=int((step < 0).sum()),
                zero_steps_in_file_order=int((step == 0).sum()),
                monotone_blocks=blocks)


def cadence_report(ts_sorted_unique, unit_scale):
    dt = np.diff(ts_sorted_unique)/unit_scale
    median = float(np.median(dt))
    on_grid = bool(np.all(np.isclose(dt/median, np.round(dt/median), atol=1e-6)))
    gaps = dt[dt > median]
    span = (ts_sorted_unique[-1]-ts_sorted_unique[0])/unit_scale
    return dict(median_dt_seconds=median, mean_dt_seconds=float(dt.mean()),
                min_dt_seconds=float(dt.min()), max_dt_seconds=float(dt.max()),
                share_dt_equal_median=float((dt == median).mean()),
                all_dt_multiples_of_median=on_grid,
                span_seconds=float(span), span_days=float(span/86400),
                grid_slots=int(round(span/median))+1,
                completeness=float(len(ts_sorted_unique)/(round(span/median)+1)),
                gap_count=int(len(gaps)), gap_total_seconds=float(gaps.sum()),
                gap_max_seconds=float(gaps.max()) if len(gaps) else 0.0,
                largest_gaps_seconds=sorted(map(float, gaps))[::-1][:10])


def book_report(frame):
    bid_p = frame[[f"bid_price_{i}" for i in LEVELS]].to_numpy(np.float64)
    ask_p = frame[[f"ask_price_{i}" for i in LEVELS]].to_numpy(np.float64)
    bid_q = frame[[f"bid_qty_{i}" for i in LEVELS]].to_numpy(np.float64)
    ask_q = frame[[f"ask_qty_{i}" for i in LEVELS]].to_numpy(np.float64)
    mid = (bid_p[:, 0]+ask_p[:, 0])/2
    spread = ask_p[:, 0]-bid_p[:, 0]
    return mid, dict(
        nan_cells=int(frame.isna().sum().sum()),
        prices_positive=bool((bid_p > 0).all() and (ask_p > 0).all()),
        quantities_nonnegative=bool((bid_q >= 0).all() and (ask_q >= 0).all()),
        zero_quantity_cells=int((bid_q == 0).sum()+(ask_q == 0).sum()),
        crossed_books=int((bid_p[:, 0] > ask_p[:, 0]).sum()),
        touching_books=int((bid_p[:, 0] == ask_p[:, 0]).sum()),
        bid_depth_strictly_descending=bool((np.diff(bid_p, axis=1) < 0).all()),
        ask_depth_strictly_ascending=bool((np.diff(ask_p, axis=1) > 0).all()),
        mid=dict(min=float(mid.min()), max=float(mid.max()), mean=float(mid.mean()),
                 std=float(mid.std())),
        spread=dict(min=float(spread.min()), median=float(np.median(spread)),
                    p99=float(np.percentile(spread, 99)), max=float(spread.max())))


def target_report(ts, mid, unit_scale, cadence, max_gap, tolerance, history_rows, stride_rows):
    """Valid-sample count and the E0 baseline in BOTH raw price and log return."""
    horizons = np.array(HORIZONS, np.int64)*int(unit_scale)
    bad = np.r_[0, np.cumsum(np.diff(ts)/unit_scale > max_gap, dtype=np.int64)]
    origins = np.arange(history_rows-1, len(ts), stride_rows, dtype=np.int64)
    wanted = ts[origins, None]+horizons
    target = np.searchsorted(ts, wanted, side="left")
    safe = np.minimum(target, len(ts)-1)
    overshoot = ts[safe]-wanted
    valid = ((target < len(ts)).all(1)
             & ((overshoot >= 0) & (overshoot <= tolerance*unit_scale)).all(1)
             & (bad[safe[:, -1]]-bad[origins-history_rows+1] == 0))
    origins, target = origins[valid], target[valid]
    origin_mid, target_mid = mid[origins][:, None], mid[target]
    price_error, log_return = target_mid-origin_mid, np.log(target_mid/origin_mid)
    return dict(history_rows=history_rows, stride_rows=stride_rows, max_gap_seconds=max_gap,
                target_tolerance_seconds=tolerance, candidate_origins=int(valid.size),
                valid_samples=int(valid.sum()), valid_share=float(valid.mean()),
                e0_rmse_price=[float(np.sqrt((price_error[:, k]**2).mean())) for k in range(3)],
                e0_mae_price=[float(np.abs(price_error[:, k]).mean()) for k in range(3)],
                e0_rmse_log_return=[float(np.sqrt((log_return[:, k]**2).mean())) for k in range(3)],
                target_mid_std=[float(target_mid[:, k].std()) for k in range(3)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="reports/eda")
    p.add_argument("--max-gap-seconds", type=float, default=None,
                   help="Default: the measured median cadence (one grid step).")
    p.add_argument("--target-tolerance-seconds", type=float, default=None)
    p.add_argument("--history-seconds", type=int, default=600)
    args = p.parse_args()

    path = Path(args.csv)
    frame = pd.read_csv(path)
    column, unit = timestamp_column(frame.columns)
    scale = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[unit]
    ts_raw = frame[column].to_numpy(np.int64)

    ordering = ordering_report(ts_raw)
    order = np.argsort(ts_raw, kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    ts = frame[column].to_numpy(np.int64)
    duplicate = np.r_[False, np.diff(ts) == 0]
    feature_columns = [c for c in frame.columns if c != column]
    identical = sum(bool(np.array_equal(frame.loc[i-1, feature_columns].to_numpy(),
                                        frame.loc[i, feature_columns].to_numpy()))
                    for i in np.flatnonzero(duplicate))
    frame = frame[~duplicate].reset_index(drop=True)
    ts = frame[column].to_numpy(np.int64)

    mid, book = book_report(frame)
    cadence = cadence_report(ts, scale)
    median = cadence["median_dt_seconds"]
    max_gap = args.max_gap_seconds if args.max_gap_seconds is not None else median
    tolerance = (args.target_tolerance_seconds
                 if args.target_tolerance_seconds is not None else median)

    resolved = {}
    for seconds in (60, 120, 180, 300, 600, 900):
        resolved[str(seconds)] = dict(history_rows=math.ceil(seconds/median),
                                      stride_seconds=seconds/6,
                                      stride_rows=max(1, int(math.floor((seconds/6)/median+0.5))))
    history_rows = math.ceil(args.history_seconds/median)
    stride_rows = max(1, int(math.floor((args.history_seconds/6)/median+0.5)))

    report = dict(
        source=dict(path=str(path.resolve()), sha256=sha256(path), size_bytes=path.stat().st_size,
                    rows_in_file=int(len(ts_raw)), rows_after_sort_and_dedup=int(len(ts)),
                    timestamp_column=column, timestamp_unit=unit,
                    has_segment_column=bool("segment_id" in frame.columns)),
        ordering=ordering | dict(duplicate_timestamp_rows=int(duplicate.sum()),
                                 duplicate_rows_with_identical_payload=int(identical)),
        interval=dict(first=str(pd.Timestamp(ts[0], unit=unit, tz="UTC")),
                      last=str(pd.Timestamp(ts[-1], unit=unit, tz="UTC"))),
        cadence=cadence, book=book,
        resolved_sampling=resolved,
        gap_rule=dict(measured_cadence_seconds=median,
                      bad_edges_at_2s=int((np.diff(ts)/scale > 2.0).sum()),
                      bad_edges_at_cadence=int((np.diff(ts)/scale > median).sum()),
                      total_edges=int(len(ts)-1)),
        targets=target_report(ts, mid, scale, median, max_gap, tolerance, history_rows, stride_rows))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}.json")


if __name__ == "__main__":
    main()
