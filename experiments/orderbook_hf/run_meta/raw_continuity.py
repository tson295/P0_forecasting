# Generator of raw_archive_continuity.json (run from repo root with /home/ubuntu/venv-ob/bin/python).
# Read-only raw-archive continuity evidence for DATA_REPORT (no replay, no model).
import duckdb, json, pandas as pd
R = "data/orderbook/hf_crypto_lob_stream"
D = [f"{R}/depth/binance/BTCUSDT/2026-06.parquet", f"{R}/depth/binance/BTCUSDT/2026-07.parquet"]
S = [f"{R}/snapshots/binance/BTCUSDT/2026-06.parquet", f"{R}/snapshots/binance/BTCUSDT/2026-07.parquet"]
c = duckdb.connect()
c.execute("""CREATE TEMP TABLE msg AS SELECT timestamp_ms ts, first_update_id f, last_update_id l, count(*) n
    FROM read_parquet(?) WHERE exchange='binance' AND asset='BTCUSDT' GROUP BY 1,2,3""", [D])
c.execute("""CREATE TEMP TABLE o AS SELECT *, lag(l) OVER w pl, lag(ts) OVER w pts FROM msg WINDOW w AS (ORDER BY l, f, ts)""")
# A "run" = maximal ID-contiguous message sequence (f <= prev_l + 1) with timestamp steps <= 10 s.
c.execute("""CREATE TEMP TABLE r AS SELECT *, sum(CASE WHEN pl IS NULL OR f > pl + 1 OR ts - pts > 10000 OR ts < pts THEN 1 ELSE 0 END)
    OVER (ORDER BY l, f, ts ROWS UNBOUNDED PRECEDING) run FROM o""")
runs = c.execute("SELECT run, min(ts) t0, max(ts) t1, count(*) msgs, min(f) f0, max(l) l1 FROM r GROUP BY run ORDER BY run").df()
runs["dur_s"] = (runs.t1 - runs.t0) / 1000
gaps = (runs.t0.shift(-1) - runs.t1).dropna() / 1000
q = [0, .01, .05, .25, .5, .75, .95, .99, 1]
out = {"messages": int(runs.msgs.sum()), "runs": len(runs),
       "run_duration_s_quantiles": dict(zip(map(str, q), runs.dur_s.quantile(q).round(3).tolist())),
       "gap_between_runs_s_quantiles": dict(zip(map(str, q), gaps.quantile(q).round(3).tolist())),
       "sum_run_duration_h": round(runs.dur_s.sum() / 3600, 3),
       "archive_span_h": round((runs.t1.max() - runs.t0.min()) / 3.6e6, 3),
       "first_utc": str(pd.to_datetime(runs.t0.min(), unit="ms", utc=True)), "last_utc": str(pd.to_datetime(runs.t1.max(), unit="ms", utc=True)),
       "longest_run_s": float(runs.dur_s.max()), "runs_ge_240s": int((runs.dur_s >= 240).sum()), "runs_ge_600s": int((runs.dur_s >= 600).sum())}
runs["start_minute"] = pd.to_datetime(runs.t0, unit="ms", utc=True).dt.minute
out["run_start_minute_top"] = runs.start_minute.value_counts().head(12).to_dict()
snap = c.execute("SELECT timestamp_ms ts, last_update_id s FROM read_parquet(?) GROUP BY 1,2 ORDER BY 1", [S]).df()
rows = []
for ts, s in snap.itertuples(index=False):
    bridge = c.execute("SELECT ts, f, l FROM msg WHERE f <= ? AND l >= ? ORDER BY l LIMIT 1", [s + 1, s + 1]).fetchone()
    nxt = c.execute("SELECT ts, f, l FROM msg WHERE l > ? ORDER BY l LIMIT 1", [s]).fetchone()
    inrun = runs[(runs.t0 <= ts) & (runs.t1 >= ts)]
    rows.append({"snapshot_utc": str(pd.to_datetime(ts, unit="ms", utc=True)), "last_update_id": int(s),
                 "bridging_msg_exists": bridge is not None,
                 "bridging_msg_dt_s": None if bridge is None else (bridge[0] - ts) / 1000,
                 "next_msg_dt_s": None if nxt is None else (nxt[0] - ts) / 1000,
                 "next_msg_missing_ids": None if nxt is None else int(nxt[1] - s - 1),
                 "inside_depth_run": len(inrun) > 0,
                 "run_remaining_s": float((inrun.t1.iloc[0] - ts) / 1000) if len(inrun) else None})
out["snapshots"] = rows
json.dump(out, open("experiments/orderbook_hf/run_meta/raw_archive_continuity.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "snapshots"}, indent=1, default=str))
print(pd.DataFrame(rows).to_string())
