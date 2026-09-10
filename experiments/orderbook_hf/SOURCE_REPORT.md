# SOURCE_REPORT — tìm nguồn BTCUSDT Binance Spot L2 thay thế (2026-09-10)

Tìm chủ động theo goal (tối đa ~60 phút), 3 vòng: **17:39–18:11 UTC**; **~18:33–18:36 UTC** (sau checker lượt 2);
**~18:54–19:02 UTC** (sau stop-hook: tag/full-text HF, Zenodo, GitHub, AWS Open Data) — tổng ~46 phút; mốc lấy từ lệnh
`date`, mtime evidence và giờ commit. Chỉ đọc metadata/schema/cột qua HTTP, không tải archive lớn, không trả phí,
không đăng ký/xin quyền. Evidence trong `run_meta/source_search/`; output truy vấn vòng 2–3 được chạy lại read-only và
lưu sau checker cuối run (F-07), file ghi thời điểm chạy lại.

## Yêu cầu tối thiểu theo phương pháp cố định

- BTCUSDT **Binance Spot**, L2 top 10 mỗi phía với giá/quantity gốc và timestamp; snapshot + diff liên tục,
  hoặc full snapshot với nhịp ≤ 10 s (age guard `max_price_age_seconds`).
- Coverage lịch ≥ 30 ngày cho 1 fold (FIT 21 + gap 6 + VAL 3), 58 ngày cho 5 fold.
- Common origin mask gồm mọi family/horizon: context giá TimesFM 512 điểm cách h ⇒ span (512−1)×h = 8,52 / 17,03 /
  **25,55 h** trước origin, cộng nhãn h ⇒ segment liên tục ≥ 8,53 / 17,07 / **25,6 h** (h60/h120/h180), trong cả FIT
  và VAL. Không giảm context/gap/FIT để ép data chạy.
- Truy cập miễn phí/công khai hoặc đã có quyền; giấy phép cho phép dùng nghiên cứu.

## Ứng viên và kết luận

| # | Nguồn | Revision / hash | Access / license | Thị trường, schema, thời gian | Coverage đo được | Kết luận (evidence) |
|---|---|---|---|---|---|---|
| 1 | HF `MaximumLeverage/crypto-lob-stream` | `873f31e7…` (1 commit) | public, MIT | Spot diff + snapshot | 1.392 run ≈300 s (5 phút/giờ), 1/38 snapshot nối được, segment tốt nhất 300,6 s | **Loại** — lỗi flush-overwrite upstream (`../DATA_REPORT.md`, `pinned_hf_card_873f31e7.md`) |
| 2 | HF `Goooddy/crypto-lob-stream` | `2b8c544ae5b4` (2026-09-01) | public, MIT | cùng archive + `2026-08.parquet` 311.202.430 B | card nguồn: tháng 8 còn ~52%, cùng lỗi collector v1 | **Loại** — mirror/cùng archive lỗi (`goooddy_card_main_2026-09-01.md`, `../run_meta/upstream_evidence.md`) |
| 3 | HF `predict-quant/binance-spot-orderbook` | `bf8ffb206841a11a65e9ebb43126ea0718cf821b` | public, **không card/license** | Spot (tick 0,01, update ID ~9,2–9,4e10); dòng `depthUpdate` (E/U/u, JSON bids/asks, 100 ms) + dòng `snapshot` 20 level **không có timestamp** | 18 file (1 rỗng) 2026-04-10 → 05-21, 0,926 GB; 650 run ID/time-liên tục, tổng 211,2 h, median 0,03 h, **dài nhất 22,64 h**; run ≥ 8,53/17,07/25,6 h: 3/2/**0**. Theo luật fold/context pipeline: 2 fold lịch; context VAL tốt nhất 7,41–7,51 h < 8,52 h ở mọi h; FIT h180 tốt nhất 22,59 h < 25,55 h | **Loại**: common mask rỗng; snapshot 20 level, không timestamp; không license (`predict_quant_spot_btcusdt_tree.json`, `pq_spot_scan.py`/`pq_spot_probe2.out`, `pq_spot_runs.py`/`.csv`/`.out`, `pq_spot_folds.py`/`.out`) |
| 4 | HF `predict-quant/binance-future-orderbook` | — | public | USDⓈ-M futures depth20 | — | **Loại** — futures (`hf_search_rounds2_3.txt`) |
| 5 | HF `payamdavaee/depth_snapshot` | `2d0e1c8fba99…` | public, không license | 1.000 level, 1.439 snapshot/ngày, bước trung vị 60,1 s, spread 0,1 ⇒ tick futures | 205 file 2026-02-17 → 09-09 (2,0 GB) | **Loại** — không phải Spot; nhịp 60 s > 10 s (`payamdavaee_btcusdt_tree.json`, `payamdavaee_probe.py`/`.out`) |
| 6 | HF `Lazy108/binance-polymarket-orderflow` | `f948a57b7fa7b80c1e426846cb61857c0230dabc` | **gated manual**, CC-BY-4.0 | mô tả API: "Binance spot L2 depth (WS snapshot)", 1 s, 20 level, BTC/ETH/SOL, 2026-08-07 onward | BTC 22 file 2026-08-07 → 08-29 (1.154.054.644 B) | **Loại** — phải xin quyền; 22 ngày < 30. Gần phương pháp nhất về dạng snapshot, liên quan phương án 1 (`lazy108_dataset_api.json`, `lazy108_btc_depth_tree.json`, `lazy108_README.md` bị hạn chế) |
| 7 | HF `rogerdehe/mktdata-binance-2026` (+ `-lbank`, `-lighter`) | `497445537d0d…` | public, MIT | USDT **perpetual** L2 deltas + snapshot mỗi ~60 s | 2026-07 → 09 | **Loại** — futures/sàn khác (`rogerdehe_mktdata_binance_2026_api.json`, `hf_search_rounds2_3.txt`) |
| 8 | HF `delmiron27/*binance-futures*`, `dataforge-labs/*` | — | public | Binance futures recorder/cryptolake; perps/options | — | **Loại** — futures/phái sinh (`hf_search_rounds2_3.txt`, `hf_tag_search_round3.txt`) |
| 9 | HF `mad0g4/l2_orderbook_binance` | `99bee8b8…` | public nhưng file `.jsonl.zst.enc` | mã hóa | 48,7 GB | **Loại** — không giải mã được (`hf_tag_search_round3.txt`) |
| 10 | HF `yinelon/crypto_lob_3m`, `_10m`, `_2y`; `mrochk/binance` | — | gated (restricted / auto — cần đăng nhập HF, không có token) | `.npy` theo symbol (có `1000PEPEUSDT` ⇒ futures); mrochk không đọc được | — | **Loại** / không truy cập (`hf_search_rounds2_3.txt`) |
| 11 | HF `alfredojrc/mercury-lob-history-v1`, `deusmos/cbb26-*`, `PXIN/fracture-crypto-l2-enriched-v2` | — | public | feature/tensor/dollar bar đã chuẩn hóa hoặc gộp (`bid_p_norm_*`; slab 10 s; đặc trưng L2 gộp), không L2 gốc Binance Spot | — | **Loại** — không có L2 gốc (`hf_tag_search_round3.txt`; PXIN theo quét của checker F-06) |
| 12 | HF `AdamAtractor/*`, `THULab/crypto_orderbook_30lvl`, `Barthel/variouscryptodata`, `asiletto81/hl_btc`, `AisotTechnologies/aisot_btc_lob_trades`, `LeonardoBerti/TRADES-LOB`, `peernagy/lob_bench`, `walkacross/orderbook` | — | public/CC-BY/gated | Hyperliquid, Polymarket, Bitstamp 2018, cổ phiếu/A-share | — | **Loại** — sai sàn/thị trường (`hf_tag_search_round3.txt`; Aisot/LOBSTER/A-share theo quét của checker F-06) |
| 13 | HF `ibrahimdaud/binance-btcusdt`, `tmmycruise/autoresearch-crypto-data` | — | public | Binance futures `bookDepth` theo dải %, mirror data.binance.vision | — | **Loại** — futures / không phải level L2 (`hf_tag_search_round3.txt`) |
| 14 | HF klines/trades/OHLCV: `jescy525/*`, `TommyKwok/*`, `edzhu/*`, `hanhvn/*`, `quant-iota/*`, `maherdik/binance-crypto-btcusdt-*`, `trade2rich/binance`, `3ltrashpanda`/`KEDevO` `crypto-market-datasets`, `commanderzee/1s-crypto-data`, `linxy/CryptoCoin`, `jponfiru/*`, `alexmindustry/*`, `duonlabs/apogee`, `mamoth/*`, `WinkingFace/*`, `Sierra-Arn/*`, `shanaka95/BTCUSDT`, `sheganinans/BTCUSD`; rỗng: `sheng9571/crypto-spot-orderbook` | — | public | klines / trades / aggTrades / funding / chuỗi giá | — | **Loại** — không có L2 (`hf_tag_search_round3.txt`, `hf_search_rounds2_3.txt`) |
| 15 | HF `rfab85/crypto-5s-market-data-adausdc-sample`, `rameez543/*`, `Pltrr/*`, `predict-quant/poly-*` | — | public | ADAUSDC; Polymarket/Kalshi; Binance BBO (L1) | — | **Loại** — sai symbol / không phải L2 Binance Spot (`hf_tag_search_round3.txt`) |
| 16 | Binance Vision `data.binance.vision` | S3 listing | public | Spot chỉ có aggTrades/klines/trades; futures có bookDepth/bookTicker | — | **Loại** — không có Spot L2 (`binance_vision_spot_daily_types.xml`) |
| 17 | Binance historical order book (`/sapi` S_DEPTH/T_DEPTH) | — | cần API key + duyệt | — | — | **Không truy cập được** |
| 18 | Tardis.dev datasets | — | không key: chỉ **ngày đầu mỗi tháng** (ngày khác HTTP 401 "only … the first day of each month") | Binance Spot `incremental_book_L2`, `book_snapshot_25` | 1 ngày/tháng ⇒ tối đa 24 h liên tục | **Loại** — < 25,6 h, không có 30 ngày liên tục; bản đầy đủ cần trả phí (`tardis_check.txt`) |
| 19 | Kaggle | — | cần credential (không có trên instance) | — | — | **Không truy cập được** |
| 20 | Crypto Chassis API; CryptoDataDownload | — | — | Chassis không phản hồi; trang Binance của CDD không có order book/depth | — | **Loại** (không lưu artifact: phản hồi rỗng / trang không có mục depth) |
| 21 | Crypto Lake sample `s3://sample.crypto.lake` (anonymous) | listing S3 | free sample "for testing" | Binance BTC-USDT `book` và `book_delta_v2` | `book` 3 ngày (2022-10-01..03), `book_delta_v2` 3 ngày (2024-04-01..03) | **Loại** — < 30 ngày (`cryptolake_sample_book_btcusdt.xml`, `cryptolake_sample_book_delta_btcusdt.xml`) |
| 22 | Zenodo `10.5281/zenodo.20046390` | `btcusdt_lob_oct2023.tar.gz`, 308.618.431 B, md5 `58507a0f…` | CC-BY-4.0 / CC-BY-NC-4.0; disclaimer "strictly for academic peer review… non-commercial" | Binance Spot REST `/api/v3/depth`, 100 level, 5 s/snapshot | **21 ngày liên tục** 2023-10-01 → 21 | **Loại** — 21 < 30 ngày ⇒ 0 fold với FIT 21/gap 6/VAL 3 (`zenodo_20046390_datacite.json`, `zenodo_20046390_files.json`) |
| 23 | Zenodo `10.5281/zenodo.10600374` | `order-book-data.zip` md5 `f1c7535f…` | CC-BY-4.0 | 6 sàn, Binance BTC-USDT 2022: snapshot **mỗi giờ** (744/tháng), giá float32 | 12 tháng | **Loại** — nhịp 60 phút, mất độ chính xác giá (`zenodo_10600374_datacite.json`, `zenodo_10600374_listing.txt`) |
| 24 | Zenodo `10.5281/zenodo.10215364` ("Limits to arbitrage for blockchain-based assets") | file list qua Zenodo API | CC0 | `best_bids_n_asks.rds` (L1) + dữ liệu arbitrage dẫn xuất | — | **Loại** — L1/dẫn xuất (`zenodo_10215364_files.json`) |
| 25 | Zenodo 8349603 / 11048480 / 21617204 / 15080493; figshare; Harvard Dataverse | — | — | DAX stocks, code, Bybit features, dữ liệu khí hậu; figshare/Dataverse không có kết quả phù hợp | — | **Loại** (`zenodo_{8349603,11048480,21617204,15080493}_{files,datacite}.json`; figshare/Dataverse không lưu artifact) |
| 26 | AWS Open Data Registry (git tree 1.203 dataset YAML) | — | — | không có dataset crypto/order book | — | **Loại** (`aws_open_data_registry_datasets.txt`) |
| 27 | GitHub repo search (dom-collector, binance-LOB, binance-lob-capture, …) | — | — | công cụ thu thập, repo ≤ 15 MB, không kèm dữ liệu nhiều tuần | — | **Loại** (`github_search_round3.txt`) |

## Kết luận

Không có nguồn miễn phí/đã có quyền nào đáp ứng đồng thời Binance Spot L2, ≥ 30 ngày lịch và segment liên tục
≥ 25,6 h trong FIT/VAL. Ba ứng viên gần nhất:

- `predict-quant/binance-spot-orderbook`: đúng Spot diff 100 ms nhưng collector đứt 20–50 lần/ngày; run dài nhất
  22,64 h; VAL fold lịch chỉ có context ≤ 7,51 h; snapshot không có timestamp và chỉ 20 level; không có license.
- Zenodo 20046390: đúng Spot, 100 level, 5 s, liên tục nhưng chỉ 21 ngày < 30 ngày cho một fold; chỉ dùng phi thương mại.
- `Lazy108/binance-polymarket-orderflow`: Spot WS snapshot 1 s/20 level nhưng gated manual và chỉ 22 ngày.

Nguồn "project đã có quyền dùng" (kiểm lúc ~19:21 UTC, chỉ in tên, không in giá trị): instance không có biến môi trường
hay file credential của nhà cung cấp dữ liệu nào (Tardis, Kaiko, CoinAPI, Crypto Lake/AWS, Kaggle, HF token, Binance API);
`CONTAINER_API_KEY` chỉ là khóa quản lý container Vast; repo không khai báo key hay nguồn trả phí nào. Không có đường truy cập
đã được cấp quyền.

Không chọn nguồn nào, không download/prepare/train thêm. Quyết định cần user: danh sách duy nhất ở `RUN_REPORT.md` §5.
