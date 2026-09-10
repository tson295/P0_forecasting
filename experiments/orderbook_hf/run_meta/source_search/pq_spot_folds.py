# Fold/context check for predict-quant spot BTCUSDT using pq_spot_runs.csv (ID/time-contiguous runs as segment proxies)
# and the pipeline's fold/context rules (src_OB/data.py Data.folds, select_origins): FIT 21 d, gap 6 d, VAL 3 d, step 7 d,
# max 5 folds; common price context (512-1) x h inside one segment, FIT context not before train_start, label t+h in segment.
# Upper bound only: real replay could only shorten segments (e.g. runs without a snapshot anchor).
import pandas as pd
DAY = 86400 * 1000  # ms
runs = pd.read_csv("pq_spot_runs.csv")
start, end = int(runs.t0.min()), int(runs.t1.max()) + 1
required = (21 + 6 + 3) * DAY
count = min(5, max(0, int((end - start - required) // (7 * DAY)) + 1))
print(f"coverage {pd.to_datetime(start, unit='ms')} -> {pd.to_datetime(end, unit='ms')}; folds by calendar: {count}")
for n in range(count):
    val_end = end - (count - 1 - n) * 7 * DAY
    val_start = val_end - 3 * DAY
    train_end = val_start - 6 * DAY
    train_start = train_end - 21 * DAY
    print(f"fold{n + 1}: FIT {pd.to_datetime(train_start, unit='ms')} -> {pd.to_datetime(train_end, unit='ms')}; "
          f"VAL {pd.to_datetime(val_start, unit='ms')} -> {pd.to_datetime(val_end, unit='ms')}")
    for h in (60, 120, 180):
        need_h = (512 - 1) * h / 3600
        hms = h * 1000
        # Best context for an origin t: t - segment start, with t + h inside the run and inside the window.
        fit = [(min(t1 - hms, train_end - hms) - max(t0, train_start)) / 3.6e6 for t0, t1 in zip(runs.t0, runs.t1)
               if min(t1 - hms, train_end - hms) > max(t0, train_start)]
        val = [(min(t1 - hms, val_end - hms) - t0) / 3.6e6 for t0, t1 in zip(runs.t0, runs.t1)
               if min(t1 - hms, val_end - hms) >= max(t0, val_start)]
        print(f"  h{h}s: context needed {need_h:.2f} h | best FIT {max(fit, default=0):.2f} h | best VAL {max(val, default=0):.2f} h"
              f" | eligible: FIT {'yes' if max(fit, default=0) >= need_h else 'no'}, VAL {'yes' if max(val, default=0) >= need_h else 'no'}")
