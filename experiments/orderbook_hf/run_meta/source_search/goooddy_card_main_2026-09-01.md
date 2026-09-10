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
pretty_name: Multi-Exchange L2 Order Book, Trades & Futures Data
configs:
# AUTO:CONFIGS_START -- regenerated every publish run from pipeline_config.py. Do not hand-edit between these markers; edits here will be overwritten.
  - config_name: snapshots_binance
    data_files: "snapshots/binance/*/*.parquet"
  - config_name: snapshots
    data_files: "snapshots/binance/*/*.parquet"
  - config_name: trades_binance
    data_files: "trades/binance/*/*.parquet"
  - config_name: trades
    data_files: "trades/binance/*/*.parquet"
  - config_name: depth_binance
    data_files: "depth/binance/*/*.parquet"
  - config_name: depth
    data_files: "depth/binance/*/*.parquet"
  - config_name: snapshots_binance_futures
    data_files: "snapshots/binance_futures/*/*.parquet"
  - config_name: trades_binance_futures
    data_files: "trades/binance_futures/*/*.parquet"
  - config_name: depth_binance_futures
    data_files: "depth/binance_futures/*/*.parquet"
  - config_name: funding_binance_futures
    data_files: "funding/binance_futures/*/*.parquet"
  - config_name: liquidations_binance_futures
    data_files: "liquidations/binance_futures/*/*.parquet"
  - config_name: open_interest_binance_futures
    data_files: "open_interest/binance_futures/*/*.parquet"
  - config_name: snapshots_coinbase
    data_files: "snapshots/coinbase/*/*.parquet"
  - config_name: trades_coinbase
    data_files: "trades/coinbase/*/*.parquet"
  - config_name: depth_coinbase
    data_files: "depth/coinbase/*/*.parquet"
  - config_name: snapshots_okx
    data_files: "snapshots/okx/*/*.parquet"
  - config_name: trades_okx
    data_files: "trades/okx/*/*.parquet"
  - config_name: depth_okx
    data_files: "depth/okx/*/*.parquet"
  - config_name: snapshots_okx_swap
    data_files: "snapshots/okx_swap/*/*.parquet"
  - config_name: trades_okx_swap
    data_files: "trades/okx_swap/*/*.parquet"
  - config_name: depth_okx_swap
    data_files: "depth/okx_swap/*/*.parquet"
  - config_name: funding_okx_swap
    data_files: "funding/okx_swap/*/*.parquet"
  - config_name: liquidations_okx_swap
    data_files: "liquidations/okx_swap/*/*.parquet"
  - config_name: open_interest_okx_swap
    data_files: "open_interest/okx_swap/*/*.parquet"
  - config_name: snapshots_kraken
    data_files: "snapshots/kraken/*/*.parquet"
  - config_name: trades_kraken
    data_files: "trades/kraken/*/*.parquet"
  - config_name: depth_kraken
    data_files: "depth/kraken/*/*.parquet"
  - config_name: snapshots_bybit
    data_files: "snapshots/bybit/*/*.parquet"
  - config_name: trades_bybit
    data_files: "trades/bybit/*/*.parquet"
  - config_name: depth_bybit
    data_files: "depth/bybit/*/*.parquet"
  - config_name: snapshots_bybit_linear
    data_files: "snapshots/bybit_linear/*/*.parquet"
  - config_name: trades_bybit_linear
    data_files: "trades/bybit_linear/*/*.parquet"
  - config_name: depth_bybit_linear
    data_files: "depth/bybit_linear/*/*.parquet"
  - config_name: funding_bybit_linear
    data_files: "funding/bybit_linear/*/*.parquet"
  - config_name: liquidations_bybit_linear
    data_files: "liquidations/bybit_linear/*/*.parquet"
  - config_name: open_interest_bybit_linear
    data_files: "open_interest/bybit_linear/*/*.parquet"
# AUTO:CONFIGS_END
---

# Multi-Exchange L2 Order Book, Trades & Futures Data

Continuous Level-2 order book (depth diffs), trades, periodic full-book
snapshots, and (for perpetual futures venues) funding rate, liquidation,
and open interest data, captured with
[`crypto-lob-stream`](https://github.com/Goodie-Goody/crypto_lob_stream_pypi)
and published monthly as a contribution to open market-microstructure
research.

Full depth-of-book data is paywalled or licensed for most asset classes;
crypto is the exception, where the raw feeds are genuinely public. This
dataset exists to lower that data barrier for independent researchers
and students -- now across eight exchanges (spot and perpetual futures)
rather than one.

## Coverage

<!-- AUTO:COVERAGE_START -- regenerated every publish run from pipeline_config.py. Do not hand-edit between these markers; edits here will be overwritten. -->
| Exchange | Assets | Data from |
|---|---|---|
| binance | BTCUSDT, ETHUSDT, SOLUSDT (from 2026-06-01); XRPUSDT, DOGEUSDT, LINKUSDT, AVAXUSDT, DOTUSDT, LTCUSDT, ADAUSDT (from 2026-09-01) | — |
| binance_futures | BTCUSDT, ETHUSDT, SOLUSDT | 2026-09-01 |
| coinbase | BTC-USD, ETH-USD, SOL-USD, XRP-USD, DOGE-USD | 2026-09-01 |
| okx | BTC-USDT, ETH-USDT, SOL-USDT, XRP-USDT, DOGE-USDT | 2026-09-01 |
| okx_swap | BTC-USDT, ETH-USDT, SOL-USDT | 2026-09-01 |
| kraken | BTC/USD, ETH/USD, SOL/USD, XRP/USD, DOGE/USD | 2026-09-01 |
| bybit | BTCUSDT, ETHUSDT, SOLUSDT, LINKUSDT, ADAUSDT | 2026-09-01 |
| bybit_linear | BTCUSDT, ETHUSDT, SOLUSDT | 2026-09-01 |
<!-- AUTO:COVERAGE_END -->

## Layout

Each table is partitioned `{prefix}/{exchange}/{asset}/YYYY-MM.parquet`.
Files are Snappy-compressed Parquet, one file per month per exchange/asset,
in `crypto-lob-stream`'s native schema (including the `exchange` column,
and `exchange_ts` alongside `timestamp_ms` on trades/depth/funding/
liquidations/open_interest as of package version 0.9.0), so its own
reconstruction helper works directly against this dataset with no
conversion step.

## Backward compatibility

This dataset covered Binance only before this release. If your code
already loads `depth`, `trades`, or `snapshots` with no exchange suffix,
it keeps working unchanged -- those names still point at Binance's data
specifically (now covering more pairs than before, as Binance's own
coverage has grown). Every exchange, including Binance, is also
available under its explicit `{prefix}_{exchange}` name (e.g.
`depth_binance`, `depth_okx`, `depth_kraken`) for anyone who wants a
specific venue.

## Reconstructing the order book

Depth rows are diffs, not a standing book. Replay a snapshot plus every
subsequent diff, pruning to your intended depth after each update, to
rebuild the book at any instant.

Heads up: a naive replay (only dropping a level when quantity hits 0)
accumulates "ghost levels" over time -- price levels that fell out of
scope but were never explicitly zeroed out, so they just sit there
looking real. There's a nasty failure mode worth knowing about: a
replay can be completely gap-free and still end up wrong.

[Oliver Zehentleitner ran an independent 25.10-hour BTCUSDT test](https://dev.to/oliverzehentleitner)
on this exact problem (gap-free, sequence-validated, no synthetic gaps
at all) found that a naive, unpruned cache grew to 20,758 bid levels and
9,116 ask levels. At the final audit against REST, only 24.09% of those
bid levels and 39.82% of the ask levels still matched. A pruned cache,
checked the same way, held steady at 1,011 bids / 1,078 asks with
87.83% / 91.74% matching REST. So don't skip the pruning step --
sequence continuity alone doesn't stop stale levels from piling up.

The `crypto-lob-stream` package handles this for you and prunes
automatically:

```bash
pip install crypto-lob-stream
```

```python
from crypto_lob_stream import reconstruct

book = reconstruct("./local_copy_of_this_dataset", exchange="binance", asset="BTCUSDT")
bids, asks = book.top(n=10)   # correctly pruned
```

If you're not using Python, the manual version is: load the nearest
snapshot before your target window, discard diffs at or before its
`last_update_id`, apply the rest in ascending `last_update_id` order
(each diff's `quantity` is the new total, not a delta -- `0.0` means
remove), and prune back to your intended depth after every single
update. That last step is the one that actually prevents the
ghost-level problem above.

See the [package README](https://github.com/Goodie-Goody/crypto_lob_stream_pypi#lob-reconstruction)
for exchange-specific default pruning depths and further detail.

## Known limitations

Binance `depth` and `trades` data for BTCUSDT/ETHUSDT/SOLUSDT from June
through August 2026 was collected by this project's original, standalone
v1 pipeline, which had a flush-naming bug causing later flushes within
the same clock hour to silently overwrite earlier ones (see the
[v0.9.2 release notes](https://github.com/Goodie-Goody/crypto_lob_stream_pypi/releases/tag/v0.9.2)
for the mechanism). That pipeline was retired without the fix ever
being applied to it, so this affects its entire operating history, not
a partial window -- the current pipeline (`crypto-lob-stream` 0.9.2+)
has never had this issue, and every other exchange's data collection
began on it directly.

Real, measured impact per month, from an audit of every published
depth/trades file for this period:

| Month | Hours affected | Estimated volume retained |
|---|---|---|
| June 2026 | ~8% | ~93% |
| July 2026 | ~14% | ~88% |
| August 2026 | ~52% | ~52% |

August specifically lost roughly half its expected `depth`/`trades`
volume for these three pairs -- worth treating as significantly
degraded for that month, not a minor gap. `snapshots` was never
vulnerable to this issue in any month.

Full explanation: see the
[v0.9.2 release notes](https://github.com/Goodie-Goody/crypto_lob_stream_pypi/releases/tag/v0.9.2).
Exact affected hours, file by file: see
[limitations_section.md](./limitations_section.md).

## Known gaps

July 2026: capture paused 2026-07-05 20:56 to ~21:39 UTC (43 min) for
Binance BTCUSDT/ETHUSDT/SOLUSDT, across depth and trades, due to a host
restart. A fresh snapshot was written on reconnect, so books before and
after the gap reconstruct cleanly -- treat the gap as a hard reset and
do not replay a diff sequence across it. (This is unrelated to the
flush-naming issue above -- see the Known limitations section for that
one specifically.)

## Where this comes from

Collected with [`crypto-lob-stream`](https://github.com/Goodie-Goody/crypto_lob_stream_pypi)
(GitHub, MIT licensed, same as this dataset), a package for streaming L2
order book, trade, funding, liquidation, and open-interest data off
public exchange WebSocket feeds straight to Parquet. Issues and PRs
welcome on the repo.
