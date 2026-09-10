# Generator of pq_spot_runs.out / pq_spot_runs.csv: ID/time-contiguous runs of predict-quant spot BTCUSDT
# across files (break: U > previous u + 1, step > 10 s or timestamp reversal), read-only over HTTPS.
# `anchored`: some snapshot lastUpdateId S lies in [first U - 1, last u) of the run, so replay could start inside it.
import duckdb, pandas as pd
REV = "bf8ffb206841a11a65e9ebb43126ea0718cf821b"
base = f"https://huggingface.co/datasets/predict-quant/binance-spot-orderbook/resolve/{REV}/"
files = """BTCUSDT/2026-04-10_BTCUSDT_depth20.parquet BTCUSDT/2026-04-11_BTCUSDT_depth20.parquet BTCUSDT/2026-04-12_BTCUSDT_depth20.parquet
BTCUSDT/2026/04/2026-04-20_BTCUSDT_depth20.parquet BTCUSDT/2026/04/2026-04-21_BTCUSDT_depth20.parquet BTCUSDT/2026/04/2026-04-22_BTCUSDT_depth20.parquet
BTCUSDT/2026/04/2026-04-23_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-07_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-08_BTCUSDT_depth20.parquet
BTCUSDT/2026/05/2026-05-09_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-14_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-15_BTCUSDT_depth20.parquet
BTCUSDT/2026/05/2026-05-17_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-18_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-19_BTCUSDT_depth20.parquet
BTCUSDT/2026/05/2026-05-20_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-21_BTCUSDT_depth20.parquet""".split()
c = duckdb.connect(); c.execute("INSTALL httpfs; LOAD httpfs;")
urls = [base + f for f in files]
c.execute("""CREATE TABLE d AS SELECT CAST(E_1 AS BIGINT) et, CAST(U AS BIGINT) uf, CAST(u_1 AS BIGINT) ul FROM read_parquet(?) WHERE e = 'depthUpdate'""", [urls])
c.execute("""CREATE TABLE s AS SELECT CAST(lastUpdateId AS BIGINT) l FROM read_parquet(?) WHERE e = 'snapshot'""", [urls])
runs = c.execute("""WITH o AS (SELECT *, lag(ul) OVER w pu, lag(et) OVER w pet FROM d WINDOW w AS (ORDER BY ul, uf, et)),
    r AS (SELECT *, sum(CASE WHEN pu IS NULL OR uf > pu + 1 OR et - pet > 10000 OR et < pet THEN 1 ELSE 0 END) OVER (ORDER BY ul, uf, et ROWS UNBOUNDED PRECEDING) run FROM o)
    SELECT run, min(et) t0, max(et) t1, count(*) msgs, min(uf) f0, max(ul) l1 FROM r GROUP BY run ORDER BY run""").df()
runs["hours"] = (runs.t1 - runs.t0) / 3.6e6
snaps = c.execute("SELECT l FROM s ORDER BY l").df().l.to_numpy()
runs["anchored"] = [bool(((snaps >= f0 - 1) & (snaps < l1)).any()) for f0, l1 in zip(runs.f0, runs.l1)]
runs["t0u"] = pd.to_datetime(runs.t0, unit="ms"); runs["t1u"] = pd.to_datetime(runs.t1, unit="ms")
print("runs", len(runs), "snapshots", len(snaps), "total hours", round(runs.hours.sum(), 2))
print("run hours quantiles", runs.hours.quantile([0, .5, .9, .99, 1]).round(3).to_dict())
print(runs.sort_values("hours", ascending=False).head(15)[["t0u", "t1u", "hours", "msgs", "anchored"]].to_string())
for h, need in ((60, 512 * 60 / 3600), (120, 512 * 120 / 3600), (180, 512 * 180 / 3600)):
    print(f"h{h}: runs >= {need:.2f} h:", int((runs.hours >= need).sum()))
runs.drop(columns=["t0u", "t1u"]).to_csv("pq_spot_runs.csv", index=False)
