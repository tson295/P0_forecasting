# EDA — BTC_L10_gate_1y.csv

Reproduce with `python scripts/eda.py --csv BTC_L10_gate_1y.csv --history-seconds 490 --output reports/eda_gate_1y`. Every number below comes from `reports/eda_gate_1y.json`.

## Verdict

The file **is** a full year of L10 book snapshots on an exact 10-second grid, but it arrives
**out of chronological order** and carries nine duplicated timestamps. Both are repaired
explicitly and recorded; neither is fixed silently.

## What the file is

| | |
|---|---|
| SHA256 | `6d8f82fbf3d0d0078b22c44cf5a06c3d82259ca16c71b16bde9b6632a8fbe6df` |
| Size | 1.07 GiB |
| Rows in file | 3,139,606 |
| Rows after sort + dedup | 3,139,597 |
| Timestamp column | `timestamp_us` (us) |
| `segment_id` column | absent |
| Interval | 2025-09-17 00:00:00+00:00 -> 2026-09-16 23:59:50+00:00 |
| Span | 365.00 days |

## Ordering: two blocks, concatenated backwards

File order is monotonic: **False** (1 backwards step).
The file is two monotone blocks:

| Block | Rows | First | Last |
|---|---:|---|---|
| 1 | 64,080 (`0..64079`) | 2026-07-01 00:00:00+00:00 | 2026-07-08 10:59:50+00:00 |
| 2 | 3,075,526 (`64080..3139605`) | 2025-09-17 00:00:00+00:00 | 2026-09-16 23:59:50+00:00 |

Block 1 is a 7.46-day chunk that sits **inside** block 2's range and fills block 2's only
multi-day hole exactly — zero timestamp overlap between them. Sorting yields one continuous series.

There are also **9 duplicated timestamps**, and **0 of them carry identical payloads** — they are two
different snapshots sharing one timestamp, not harmless copies, so the drop policy is explicit
(`keep_first`).

## Cadence: an exact 10-second grid

| | |
|---|---|
| Median dt | 10.0 s |
| Share of steps exactly at median | 99.999% |
| All dt multiples of the median | True |
| Gaps > 10 s | 31 |
| Total missing time | 1.62 days |
| Largest gap | 2.00 h |
| Grid completeness | 99.56% |

### This breaks the old gap rule

At 10 s cadence the previous `max_gap_seconds = 2.0` marks **3,139,596 of 3,139,596 edges bad (100.0000%)** — every sample would be rejected. At
`max_gap_seconds = 10.0` only **31** edges are bad. The gap rule and the
target tolerance are therefore rescaled 2.0 s -> 10.0 s, which is the same rule expressed in this
file's cadence.

## Book quality: clean

| Check | Result |
|---|---|
| NaN cells | 0 |
| Prices > 0 | True |
| Quantities >= 0 | True |
| Zero-quantity cells | 0 |
| Crossed books | 0 |
| Touching books (bid1 == ask1) | 0 |
| Bid depth strictly descending | True |
| Ask depth strictly ascending | True |
| Mid range | 57,847.85 -> 126,191.35 (mean 81,051.72) |
| Spread | median 0.10, p99 0.10, max 552.90 |

## History window: 60 s no longer works

The window is resolved as `ceil(history_seconds / median_dt)`, so at 10 s cadence:

| history_seconds | history_rows | stride_rows | PatchTST/ModernTCN need rows >= 16 |
|---:|---:|---:|---|
| 60 | 6 | 1 | **FAILS** |
| 120 | 12 | 2 | **FAILS** |
| 180 | 18 | 3 | OK |
| 300 | 30 | 5 | OK |
| 600 | 60 | 10 | OK |
| 900 | 90 | 15 | OK |

The original 60 s window gives only 6 snapshots, which is below `patch_length = 16` and raises in
PatchTST. **`history_seconds = 490` is used instead**: it re-resolves to `history_rows = 49` and
`stride_rows = 8`, exactly the shapes the first experiment used, so all five parameter counts stay
unchanged and only the data scales.

## Targets and the E0 baseline in raw price

At history_rows=60, stride_rows=10, max_gap=10 s, tolerance=10 s: **313,731 of 313,954** candidate origins are valid (99.93%).

| Horizon | E0 RMSE (USD) | E0 MAE (USD) | E0 RMSE (log return) | sigma(target mid) (USD) |
|---|---:|---:|---:|---:|
| 1m | 50.388 | 30.107 | 6.199e-04 | 16,579 |
| 2m | 71.682 | 43.829 | 8.806e-04 | 16,579 |
| 3m | 87.940 | 54.064 | 1.080e-03 | 16,579 |

**Warning on R2 in price space.** sigma(target mid) is about 16,600 USD while every error in this
table is around 50-90 USD, so standard R2 computed on the price level sits near 0.9999 for any
model, E0 included. It is still reported for contract completeness, but `rmse_gain_vs_e0` is the
column that carries information.
