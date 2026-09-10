# Generator of pq_spot_probe2.out: read-only per-file scan of predict-quant/binance-spot-orderbook BTCUSDT
# (column projection over HTTPS; DuckDB renames the case-colliding columns E -> E_1 and u -> u_1).
import duckdb, pandas as pd
pd.set_option("display.width", 250)
REV = "bf8ffb206841a11a65e9ebb43126ea0718cf821b"
base = f"https://huggingface.co/datasets/predict-quant/binance-spot-orderbook/resolve/{REV}/"
files = """BTCUSDT/2026-04-10_BTCUSDT_depth20.parquet BTCUSDT/2026-04-11_BTCUSDT_depth20.parquet BTCUSDT/2026-04-12_BTCUSDT_depth20.parquet
BTCUSDT/2026/04/2026-04-20_BTCUSDT_depth20.parquet BTCUSDT/2026/04/2026-04-21_BTCUSDT_depth20.parquet BTCUSDT/2026/04/2026-04-22_BTCUSDT_depth20.parquet
BTCUSDT/2026/04/2026-04-23_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-07_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-08_BTCUSDT_depth20.parquet
BTCUSDT/2026/05/2026-05-09_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-14_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-15_BTCUSDT_depth20.parquet
BTCUSDT/2026/05/2026-05-17_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-18_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-19_BTCUSDT_depth20.parquet
BTCUSDT/2026/05/2026-05-20_BTCUSDT_depth20.parquet BTCUSDT/2026/05/2026-05-21_BTCUSDT_depth20.parquet""".split()
c = duckdb.connect(); c.execute("INSTALL httpfs; LOAD httpfs;")
snap = c.execute(f"""SELECT e, E_1, lastUpdateId, U, u_1, json_array_length(bids) nb, json_array_length(asks) na
    FROM read_parquet('{base + files[-1]}') WHERE e = 'snapshot' ORDER BY lastUpdateId""").df()
print(snap.head(30).to_string())
rows = []
for f in files:
    q = f"""WITH x AS (SELECT e, CAST(E_1 AS BIGINT) et, CAST(lastUpdateId AS BIGINT) luid, CAST(U AS BIGINT) uf, CAST(u_1 AS BIGINT) ul
                       FROM read_parquet('{base + f}')),
         d AS (SELECT * FROM x WHERE e = 'depthUpdate'),
         o AS (SELECT *, lag(ul) OVER w pu, lag(et) OVER w pet FROM d WINDOW w AS (ORDER BY ul, uf, et))
         SELECT count(*), min(et), max(et), max(et - pet), quantile_cont(et - pet, 0.5), count(*) FILTER (WHERE et - pet > 10000),
                count(*) FILTER (WHERE et < pet), count(*) FILTER (WHERE uf > pu + 1), count(*) FILTER (WHERE ul <= pu),
                (SELECT count(*) FROM x WHERE e = 'snapshot'), (SELECT count(*) FROM x WHERE e NOT IN ('snapshot', 'depthUpdate') OR e IS NULL),
                min(uf), max(ul), arg_min(uf, ul), arg_max(et, ul) FROM o"""
    r = c.execute(q).fetchone()
    rows.append(dict(file=f.split('/')[-1][:10], n=r[0], first=pd.to_datetime(r[1], unit="ms"), last=pd.to_datetime(r[2], unit="ms"),
                     max_dE_s=(r[3] or 0) / 1000, med_dE_ms=r[4], gaps10s=r[5], ts_rev=r[6], id_gaps=r[7], dup_obsolete=r[8],
                     snapshots=r[9], other=r[10], first_U=r[11], last_u=r[12]))
t = pd.DataFrame(rows); print(t.to_string())
t["next_first_U"] = t.first_U.shift(-1); t["bridge_gap_ids"] = t.next_first_U - t.last_u - 1
t["bridge_gap_s"] = (t.first.shift(-1) - t["last"]).dt.total_seconds()
print(t[["file", "last_u", "next_first_U", "bridge_gap_ids", "bridge_gap_s"]].to_string())
