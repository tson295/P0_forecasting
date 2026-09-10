---
license: mit
task_categories:
  - time-series-forecasting
tags:
  - finance
  - cryptocurrency
  - order-book
  - market-microstructure
  - high-frequency
pretty_name: Binance BTC/ETH/SOL L2 Order Book & Trades
configs:
  - config_name: depth
    data_files: "depth/binance/*/*.parquet"
  - config_name: trades
    data_files: "trades/binance/*/*.parquet"
  - config_name: snapshots
    data_files: "snapshots/binance/*/*.parquet"
---

# Binance BTC/ETH/SOL - L2 Order Book, Trades & Snapshots

Continuous Level-2 order book (depth diffs), trades, and periodic full-book
snapshots for BTCUSDT, ETHUSDT, and SOLUSDT, captured from Binance's public
WebSocket feeds with [`crypto-lob-stream`](https://pypi.org/project/crypto-lob-stream/)
and published monthly as a contribution to open market-microstructure research.

Full depth-of-book data is paywalled or licensed for most asset classes; crypto
is the exception, where the raw feeds are genuinely public. This dataset exists
to lower that data barrier for independent researchers and students.

## Layout

```
depth/binance/{ASSET}/YYYY-MM.parquet        # order book diff events
trades/binance/{ASSET}/YYYY-MM.parquet       # executed trades
snapshots/binance/{ASSET}/*.parquet          # full-book anchors for replay
```

Files are Snappy-compressed Parquet, compacted to one file per month per asset,
and carry an explicit `exchange` column - i.e. the native schema of the
`crypto-lob-stream` package, so its reconstruction helper works directly.

## Reconstructing the order book

Depth rows are diffs, not a standing book. Replay a snapshot plus every
subsequent diff, pruning to your intended depth after each update, to rebuild
the book at any instant.

Heads up: a naive replay (only dropping a level when `quantity` hits `0`)
accumulates "ghost levels" over time: price levels that fell out of scope
but were never explicitly zeroed out, so they just sit there looking real.
There's a nasty failure mode worth knowing about: a replay can be
completely gap-free and still end up wrong.

Oliver Zehentleitner ran a 25.10-hour BTCUSDT test on this exact problem
(gap-free, sequence-validated, no synthetic gaps at all) and found that a
naive, unpruned cache grew to 20,758 bid levels and 9,116 ask levels. At
the final audit against REST, only 24.09% of those bid levels and 39.82%
of the ask levels still matched. A pruned cache, checked the same way,
held steady at 1,011 bids / 1,078 asks with 87.83% / 91.74% matching REST.
So don't skip the pruning step, sequence continuity alone doesn't stop
stale levels from piling up. [Full writeup here.](https://dev.to/oliverzehentleitner/your-binance-l2-order-book-can-be-gap-free-and-still-be-wrong-1kk8)

The [`crypto-lob-stream`](https://pypi.org/project/crypto-lob-stream/)
package (source: [GitHub](https://github.com/Goodie-Goody/crypto_lob_stream_pypi))
handles this for you and prunes automatically:

```python
pip install crypto-lob-stream

from crypto_lob_stream import reconstruct

# point this at the folder containing trades/, depth/, snapshots/
book = reconstruct("./", exchange="binance", asset="BTCUSDT")
bids, asks = book.top(n=10)  # best 10 levels per side, correctly pruned
```

By default it prunes Binance books to depth 1000 (that's the depth the
snapshot REST call requests. Binance itself doesn't cap the live diff
stream, so without pruning the book can genuinely drift past that over
time). Pass `max_depth=` to override, but I wouldn't disable it.

If you're not using Python, the manual version is: load the nearest
snapshot before your target window, discard diffs at or before its
`last_update_id`, apply the rest in ascending `last_update_id` order
(each diff's `quantity` is the new total, not a delta, `0.0` means
remove), and prune back to your intended depth after every single update.
That last step is the one that actually prevents the ghost-level problem
above.

## Schemas

**trades**

| Field | Type | Notes |
|---|---|---|
| timestamp_ms | int64 | Event time (Unix ms) |
| exchange | string | `binance` |
| asset | string | e.g. BTCUSDT |
| trade_id | int64 | Exchange trade ID |
| price | float64 | |
| quantity | float64 | |
| buyer_maker | bool | True if the buyer was the maker |

**depth (diff events)**

| Field | Type | Notes |
|---|---|---|
| timestamp_ms | int64 | Event/receipt time (Unix ms) |
| exchange | string | |
| asset | string | |
| side | string | bid or ask |
| price | float64 | |
| quantity | float64 | `0.0` means the level was removed |
| first_update_id | int64 | Sequence start |
| last_update_id | int64 | Sequence number for replay ordering |

**snapshots**

| Field | Type | Notes |
|---|---|---|
| timestamp_ms | int64 | Snapshot time (Unix ms) |
| exchange | string | |
| asset | string | |
| side | string | bid or ask |
| price | float64 | |
| quantity | float64 | |
| last_update_id | int64 | Snapshot anchor for diff replay |

## Coverage & limitations

- **Reconstructable from 2026-06-03 onward.** Every depth file here carries
  `first_update_id` / `last_update_id`, so the diff sequence can be replayed
  against the snapshots. Earlier data lacks sequence ids and is not published.
- Reconstruct each market on its own - never merge books across exchanges.
- Check the diff sequence for continuity before replaying across a boundary.

## Known gaps

- **July 2026:** capture paused ~2026-07-05 20:56 to ~21:39 UTC (~43 min) for
  all three symbols, across depth and trades, due to a host restart. A fresh
  snapshot was written on reconnect, so books before and after the gap
  reconstruct cleanly - treat the gap as a hard reset and do not replay a diff
  sequence across it.

## Where this comes from

Collected with [`crypto-lob-stream`](https://pypi.org/project/crypto-lob-stream/)
([GitHub](https://github.com/Goodie-Goody/crypto_lob_stream_pypi), MIT licensed,
same as this dataset), a package I built for streaming L2 order book, trade,
and funding data off public exchange WebSocket feeds straight to Parquet.
Binance/BTC/ETH/SOL is what's published here, but the package also supports
Coinbase, OKX, Kraken, and Bybit (spot and perps) if you want to collect your
own. Issues and PRs welcome on the repo.
