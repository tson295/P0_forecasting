# Upstream evidence (read-only HF API/card reads, 2026-09-10 UTC)

## Pinned repo `MaximumLeverage/crypto-lob-stream`

- `main` has a single commit: `873f31e729ae` 2026-08-13T17:34:40Z "Duplicate from Goooddy/crypto-lob-stream"
  (= the pinned `dataset_revision`). No newer revision exists on this repo.
- BTCUSDT files at `main` are byte-identical (same LFS sha256 prefixes) to the pinned download:
  depth 2026-06 (385,932,069 B, `e6339adc…`), depth 2026-07 (317,605,448 B, `4a5d581d…`),
  snapshots 2026-06 (299,002 B), snapshots 2026-07 (350,331 B).
- Card at the pinned revision describes the data as "Continuous Level-2 order book (depth diffs)" and lists only
  the 2026-07-05 20:56–21:39 UTC host-restart gap.

## Source repo `Goooddy/crypto-lob-stream` (not used by the pipeline; revision unchanged)

- Latest commits: `2b8c544ae5b4` 2026-09-01T10:03:46Z "Upload limitations_section.md", `3cb8267210bd` README,
  `6b223cb6df1a` 2026-09-01T09:28:38Z "Upload folder", … `218a57cb525f` 2026-08-02, `70f370c873f5` 2026-07-05.
- BTCUSDT depth at `main`: 2026-06 and 2026-07 identical to the pinned files, plus `2026-08.parquet`
  (311,202,430 B, `0c9c95d5…`) and snapshots `2026-08.parquet` (331,636 B). The August depth file has about the
  same size as June/July, i.e. not the ~12x volume a continuous month would have.
- Card "Known limitations" (verbatim excerpt):

> Binance `depth` and `trades` data for BTCUSDT/ETHUSDT/SOLUSDT from June through August 2026 was collected by
> this project's original, standalone v1 pipeline, which had a flush-naming bug causing later flushes within the
> same clock hour to silently overwrite earlier ones (see the v0.9.2 release notes for the mechanism). That
> pipeline was retired without the fix ever being applied to it, so this affects its entire operating history,
> not a partial window -- the current pipeline (`crypto-lob-stream` 0.9.2+) has never had this issue […]
> `snapshots` was never vulnerable to this issue in any month.

- The same card estimates "~93% / ~88% / ~52%" volume retained for June/July/August and flags only 48 (June) and
  99 (July) "narrow" hours. This does **not** match the downloaded files: measured directly, every hour holds a
  single ~300 s run of ID-contiguous depth messages starting at minute :49–:55 (1,392 runs, max 300.4 s, 115.8 h of
  1,391 h). See `raw_archive_continuity.json` and `../DATA_REPORT.md`.
- Coverage table: new pipeline (0.9.2+) venues start 2026-09-01; Binance spot BTCUSDT is listed "from 2026-06-01"
  (v1 pipeline months are the affected ones).
