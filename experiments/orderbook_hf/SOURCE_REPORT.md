# SOURCE_REPORT — tìm nguồn BTCUSDT Binance Spot L2 thay thế (2026-09-10)

Tìm chủ động theo goal (tối đa ~60 phút): vòng 1 **17:39–18:11 UTC**, vòng 2 sau checker **~18:33–18:36 UTC**
(~35 phút; mốc lấy từ lệnh `date`, mtime evidence và giờ commit). Chỉ đọc metadata/schema/cột qua HTTP, không tải archive lớn, không trả phí, không đăng ký/xin quyền.
Bằng chứng: `run_meta/source_search/` (tree/metadata đã lưu, card, listing, script + output của từng probe).

## Yêu cầu tối thiểu theo phương pháp cố định

- BTCUSDT **Binance Spot**, L2 top 10 mỗi phía với giá/quantity gốc và timestamp; snapshot + diff liên tục,
  hoặc full snapshot với nhịp ≤ 10 s (age guard `max_price_age_seconds`).
- Coverage lịch ≥ 30 ngày cho 1 fold (FIT 21 + gap 6 + VAL 3), 58 ngày cho 5 fold.
- Common origin mask gồm mọi family/horizon: context giá TimesFM 512 điểm cách h ⇒ span (512−1)×h = 8,52 / 17,03 /
  **25,55 h** trước origin, cộng nhãn h ⇒ segment liên tục ≥ 8,53 / 17,07 / **25,6 h** (h60/h120/h180), trong cả FIT
  và VAL. Không giảm context/gap/FIT để ép data chạy.
- Truy cập miễn phí/công khai hoặc đã có quyền; giấy phép cho phép dùng nghiên cứu.

## Ứng viên và kết luận

| # | Nguồn | Revision / hash | Access / license | Thị trường, schema, thời gian | Coverage đo được | Kết luận |
|---|---|---|---|---|---|---|
| 1 | HF `MaximumLeverage/crypto-lob-stream` | `873f31e7…` (1 commit) | public, MIT | Spot diff + snapshot | 1.392 run ≈300 s (5 phút/giờ), 1/38 snapshot nối được, segment tốt nhất 300,6 s | **Loại** (DATA_REPORT; lỗi flush-overwrite upstream) |
| 2 | HF `Goooddy/crypto-lob-stream` | `2b8c544ae5b4` (2026-09-01) | public, MIT | cùng archive + `2026-08.parquet` 311.202.430 B | card nguồn: tháng 8 còn ~52%, cùng lỗi collector v1 (`goooddy_card_main_2026-09-01.md`) | **Loại** (mirror/cùng archive lỗi) |
| 3 | HF `predict-quant/binance-spot-orderbook` | `bf8ffb206841a11a65e9ebb43126ea0718cf821b` | public, **không card/license** | Spot (tick 0,01, update ID ~9,2–9,4e10); dòng `depthUpdate` (E/U/u, JSON bids/asks, 100 ms) + dòng `snapshot` 20 level **không có timestamp** | 18 file (1 rỗng) 2026-04-10 → 05-21, 0,926 GB, thiếu nhiều ngày; 650 run ID/time-liên tục, tổng 211,2 h, median 0,03 h, **dài nhất 22,64 h**; run ≥ 8,53 / 17,07 / 25,6 h: 3 / 2 / **0**. Theo mask pipeline: 2 fold lịch; context VAL tốt nhất 7,44 / 7,51 h < 8,52 h (h60); context FIT tốt nhất ở h180 22,59 h < 25,55 h (`pq_spot_scan.py`, `pq_spot_runs.py`) | **Loại**: common mask rỗng; snapshot 20 level, không timestamp; không license |
| 4 | HF `predict-quant/binance-future-orderbook` | — | public | USDⓈ-M futures depth20 | — | **Loại** (futures) |
| 5 | HF `payamdavaee/depth_snapshot` | `2d0e1c8fba99…` | public, không license | 1.000 level, 1.439 snapshot/ngày, bước trung vị 60,1 s, spread 0,1 ⇒ tick futures (`payamdavaee_probe.out`) | 205 file 2026-02-17 → 09-09 (2,0 GB) | **Loại** (không phải Spot; nhịp 60 s > age guard 10 s) |
| 6 | HF `Lazy108/binance-polymarket-orderflow` | `f948a57b7fa7b80c1e426846cb61857c0230dabc` | **gated manual**, CC-BY-4.0 | card: Binance Spot L2 depth (WS snapshot), 1 s, 20 level | BTC 22 file 2026-08-07 → 08-29 | **Loại** (phải xin quyền; 22 ngày < 30). Gần phương pháp nhất về dạng snapshot, liên quan phương án (a) |
| 7 | HF `rogerdehe/mktdata-binance-2026` | `497445537d0d…` | public, MIT | USDT **perpetual** L2 deltas + snapshot mỗi ~60 s | 2026-07 → 09 | **Loại** (futures) |
| 8 | HF `delmiron27/*binance-futures*` | — | public | Binance futures recorder/cryptolake | — | **Loại** (futures) |
| 9 | HF `mad0g4/l2_orderbook_binance` | `99bee8b8…` | public nhưng file `.jsonl.zst.enc` | mã hóa | 48,7 GB | **Loại** (không giải mã được) |
| 10 | HF `yinelon/crypto_lob_3m`, `_10m`, `_2y`; `mrochk/binance` | — | gated (restricted / auto — cần đăng nhập HF, không có token) | `.npy` theo symbol (có `1000PEPEUSDT` ⇒ futures); mrochk không đọc được | — | **Loại** / không truy cập |
| 11 | HF `alfredojrc/mercury-lob-history-v1` | — | public | chỉ feature chuẩn hóa (`bid_p_norm_*`, `target_return`), không giá gốc/exchange | 2024-11 → 2025-03 | **Loại** (không có L2 gốc) |
| 12 | HF `AdamAtractor/*`, `THULab/crypto_orderbook_30lvl` | — | CC-BY-4.0 sample | Hyperliquid perp, bar 1–5 phút | 7 ngày sample | **Loại** (sai sàn/thị trường) |
| 13 | HF `ibrahimdaud/binance-btcusdt` | — | public | Binance futures `bookDepth` theo dải % | 2023 → 2026 | **Loại** (futures, không phải level L2) |
| 14 | HF `jescy525/*`, `TommyKwok/*`, `edzhu/*`, `hanhvn/*`, `quant-iota/*`, `maherdik/binance-crypto-btcusdt-*`, `trade2rich/binance`, `3ltrashpanda`/`KEDevO` `crypto-market-datasets` | — | public | klines / trades / aggTrades / funding | — | **Loại** (không có L2) |
| 15 | HF `rfab85/crypto-5s-market-data-adausdc-sample`, `rameez543/*`, `Pltrr/*`, `predict-quant/poly-*` | — | public | ADAUSDC; Polymarket/Kalshi; Binance BBO (L1) | — | **Loại** (sai symbol / không phải L2 Binance Spot) |
| 16 | Binance Vision `data.binance.vision` | S3 listing | public | Spot chỉ có aggTrades/klines/trades; futures có bookDepth/bookTicker | — | **Loại** (không có Spot L2) |
| 17 | Binance historical order book (`/sapi` S_DEPTH/T_DEPTH) | — | cần API key + duyệt | — | — | **Không truy cập được** |
| 18 | Tardis.dev datasets | — | không key: chỉ **ngày đầu mỗi tháng** (ngày khác HTTP 401 "only … the first day of each month", `tardis_check.txt`) | Binance Spot `incremental_book_L2`, `book_snapshot_25` | 1 ngày/tháng ⇒ tối đa 24 h liên tục | **Loại** (< 25,6 h, không có 30 ngày liên tục; bản đầy đủ cần trả phí) |
| 19 | Kaggle | — | cần credential (không có trên instance) | — | — | **Không truy cập được** |
| 20 | Crypto Chassis API; CryptoDataDownload | — | — | Chassis không phản hồi; trang Binance của CDD không có order book/depth | — | **Loại** |
| 21 | Crypto Lake sample `s3://sample.crypto.lake` (anonymous) | listing S3 đã lưu | free sample "for testing" | Binance BTC-USDT `book` và `book_delta_v2` | `book` 3 ngày (2022-10-01..03), `book_delta_v2` 3 ngày (2024-04-01..03) | **Loại** (< 30 ngày) |
| 22 | Zenodo `10.5281/zenodo.20046390` | `btcusdt_lob_oct2023.tar.gz`, 308.618.431 B, md5 `58507a0f…` (`zenodo_20046390_files.json`) | CC-BY-4.0 / CC-BY-NC-4.0; disclaimer "strictly for academic peer review… non-commercial" | Binance Spot REST `/api/v3/depth`, 100 level, 5 s/snapshot | **21 ngày liên tục** 2023-10-01 → 21 | **Loại**: 21 < 30 ngày ⇒ 0 fold với FIT 21/gap 6/VAL 3 |
| 23 | Zenodo `10.5281/zenodo.10600374` | `order-book-data.zip` md5 `f1c7535f…` (`zenodo_10600374_listing.txt`) | CC-BY-4.0 | 6 sàn, Binance BTC-USDT 2022: snapshot **mỗi giờ** (744/tháng), giá float32 | 12 tháng | **Loại** (nhịp 60 phút, mất độ chính xác giá) |
| 24 | Zenodo 8349603 / 11048480 / 21617204; figshare; Harvard Dataverse | — | — | DAX stocks, code, Bybit features; không có kết quả phù hợp | — | **Loại** |

## Kết luận

Không có nguồn miễn phí/đã có quyền nào đáp ứng đồng thời Binance Spot L2, ≥ 30 ngày lịch và segment liên tục
≥ 25,6 h trong FIT/VAL. Ba ứng viên gần nhất:

- `predict-quant/binance-spot-orderbook`: đúng Spot diff 100 ms nhưng collector đứt 20–50 lần/ngày; run dài nhất
  22,64 h; VAL fold lịch chỉ có context ≤ 7,5 h; snapshot không có timestamp và chỉ 20 level; không có license.
- Zenodo 20046390: đúng Spot, 100 level, 5 s, liên tục nhưng chỉ 21 ngày < 30 ngày cho một fold; chỉ dùng phi thương mại.
- `Lazy108/binance-polymarket-orderflow`: Spot WS snapshot 1 s/20 level nhưng gated manual và chỉ 22 ngày.

Không chọn nguồn nào, không download/prepare/train thêm. Quyết định tối thiểu cần user: (a) cấp nguồn có trả phí/khóa
hoặc quyền truy cập (Tardis API key, Binance historical data access, Crypto Lake/Kaiko, duyệt dataset gated…) cho
≥ 30–58 ngày Spot liên tục; (b) chấp nhận đổi phương pháp tường minh (vd. FIT ngắn hơn cho bộ 21 ngày Zenodo, hoặc
context TimesFM/h180 khác cho predict-quant); hoặc (c) thu thập mới ≥ 30/58 ngày bằng collector liên tục.
