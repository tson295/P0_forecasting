# Read-only remote probe of payamdavaee/depth_snapshot BTCUSDT (row cadence, tick, depth) for SOURCE_REPORT.
import duckdb
c = duckdb.connect(); c.execute("INSTALL httpfs; LOAD httpfs;")
u = ("https://huggingface.co/datasets/payamdavaee/depth_snapshot/resolve/2d0e1c8fba996d00f167c91a4b60c2769dde3562/"
     "btcusdt/btcusdt_depth_snapshot_2026_05_20.parquet")
print(c.execute(f"""SELECT count(*) n_rows, min(ts) first_ts, max(ts) last_ts, median(date_diff('millisecond', lag_ts, ts)) median_step_ms,
    min(spread) min_spread, median(spread) median_spread, min(bcount) min_bid_levels, min(acount) min_ask_levels
    FROM (SELECT ts, lag(ts) OVER (ORDER BY ts) lag_ts, spread, bcount, acount FROM read_parquet('{u}'))""").df().to_string())
print(c.execute(f"SELECT bids[1].price best_bid, bids[2].price bid2, asks[1].price best_ask FROM read_parquet('{u}') LIMIT 3").df().to_string())
