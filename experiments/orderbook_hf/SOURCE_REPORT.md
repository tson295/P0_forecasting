# SOURCE_REPORT — tìm nguồn BTCUSDT Binance Spot L2 thay thế (2026-09-10, ~17:40–18:40 UTC)

Tìm chủ động ~60 phút theo goal (chỉ đọc metadata/schema/cột qua HTTP, không tải archive lớn, không trả phí,
không đăng ký). Bằng chứng thô: `run_meta/source_search/` (tree/metadata đã lưu, `pq_spot_runs.csv`, probe scripts).

## Yêu cầu tối thiểu theo phương pháp cố định

- BTCUSDT **Binance Spot**, L2 top 10 mỗi phía với giá/quantity gốc và timestamp; snapshot + diff liên tục,
  hoặc full snapshot với nhịp ≤ 10 s (age guard `max_price_age_seconds`).
- Coverage lịch ≥ 30 ngày cho 1 fold (FIT 21 + gap 6 + VAL 3), 58 ngày cho 5 fold.
- Common origin mask gồm mọi family/horizon: TimesFM 512 × h ⇒ segment liên tục ≥ 8,53 h (h60), 17,07 h (h120),
  **25,6 h (h180)** trước origin, trong cả FIT và VAL. Không được giảm context/gap/FIT để ép data chạy.
- Truy cập miễn phí/công khai hoặc đã có quyền; giấy phép cho phép dùng nghiên cứu.

## Ứng viên và kết luận

| # | Nguồn | Revision / hash | Access / license | Thị trường, schema, thời gian | Coverage đo được | Kết luận |
|---|---|---|---|---|---|---|
| 1 | HF `MaximumLeverage/crypto-lob-stream` | `873f31e7…` (1 commit) | public, MIT | Spot diff + snapshot | 1.392 run ≈300 s (5 phút/giờ), 1/38 snapshot nối được, segment tốt nhất 300,6 s | **Loại** (DATA_REPORT, lỗi flush-overwrite upstream) |
| 2 | HF `Goooddy/crypto-lob-stream` | `2b8c544ae5b4` (2026-09-01) | public, MIT | cùng archive + `2026-08.parquet` 311 MB | card nguồn: tháng 8 còn ~52%, cùng lỗi v1 | **Loại** (mirror/cùng archive lỗi) |
| 3 | HF `predict-quant/binance-spot-orderbook` | `bf8ffb206841a11a65e9ebb43126ea0718cf821b` | public, **không có card/license** | Spot (tick 0,01, update ID ~9,2–9,4e10); dòng `depthUpdate` (E/U/u, JSON bids/asks, 100 ms) + dòng `snapshot` 20 level **không có timestamp** | 17 file ngày khác rỗng 2026-04-10 → 05-21 (0,93 GB), thiếu nhiều ngày; 650 run ID/time-liên tục, tổng 211,2 h, median 0,03 h, **dài nhất 22,64 h** (04-22 02:48 → 04-23 01:26); run ≥ 8,53 h: 3, ≥ 17,07 h: 2, **≥ 25,6 h: 0**; VAL của 2 fold lịch chỉ có run ≤ 7,5 h | **Loại**: common mask h180 rỗng (và VAL rỗng cả ở h60); snapshot chỉ 20 level, không timestamp |
| 4 | HF `predict-quant/binance-future-orderbook` | — | public | USDⓈ-M futures depth20 | — | **Loại** (futures) |
| 5 | HF `payamdavaee/depth_snapshot` | `2d0e1c8fba99…` | public, không license | 1 snapshot/phút (1.439 dòng/ngày), 1000 level, **tick 0,1 ⇒ futures** | 205 file 2026-02-17 → 09-09 (2,0 GB) | **Loại** (không phải Spot; nhịp 60 s > age guard 10 s) |
| 6 | HF `mad0g4/l2_orderbook_binance` | `99bee8b8…` | public nhưng file `.jsonl.zst.enc` | mã hóa | 48,7 GB | **Loại** (không giải mã được) |
| 7 | HF `yinelon/crypto_lob_3m`, `_10m`, `_2y` | — | **gated** (restricted) | `.npy` theo symbol (có `1000PEPEUSDT` ⇒ futures) | 2025-08 → 11 | **Loại** (không có quyền; không phải Spot) |
| 8 | HF `alfredojrc/mercury-lob-history-v1` | — | public | chỉ feature đã chuẩn hóa (`bid_p_norm_*`, `target_return`), không giá gốc/exchange | 2024-11 → 2025-03 | **Loại** (không có L2 gốc) |
| 9 | HF `AdamAtractor/*`, `THULab/crypto_orderbook_30lvl` | — | CC-BY-4.0 sample | Hyperliquid perp, bar 1–5 phút | 7 ngày sample | **Loại** (sai sàn/thị trường) |
| 10 | HF `ibrahimdaud/binance-btcusdt` | — | public | Binance futures `bookDepth` theo dải % | 2023 → 2026 | **Loại** (futures, không phải level L2) |
| 11 | HF `jescy525/binance-trading-272gb`, `TommyKwok/*`, `edzhu/*`, `hanhvn/*`, `quant-iota/*` | — | public | klines / trades / aggTrades | — | **Loại** (không có L2) |
| 12 | HF `rameez543/*`, `Pltrr/*`, `predict-quant/poly-*` | — | public | Polymarket/Kalshi, Binance BBO (L1) | — | **Loại** |
| 13 | Binance Vision `data.binance.vision` | S3 listing | public | Spot chỉ có aggTrades/klines/trades; futures có bookDepth/bookTicker | — | **Loại** (không có Spot L2) |
| 14 | Binance historical order book (`/sapi` S_DEPTH/T_DEPTH) | — | cần API key + duyệt | — | — | **Không truy cập được** |
| 15 | Tardis.dev datasets | — | không key: chỉ **ngày đầu mỗi tháng** (GET đầy đủ HTTP 200; range request 403) | Binance Spot `incremental_book_L2` | 1 ngày/tháng ⇒ tối đa 24 h liên tục | **Loại** (< 25,6 h; không có 30 ngày liên tục; bản đầy đủ cần trả phí) |
| 16 | Kaggle | — | cần credential (không có trên instance) | — | — | **Không truy cập được** |
| 17 | Crypto Chassis API | — | — | — | không phản hồi | **Loại** (dịch vụ không hoạt động) |
| 18 | CryptoDataDownload | — | — | trang Binance không có order book/depth | — | **Loại** |
| 19 | Crypto Lake sample `s3://sample.crypto.lake` (anonymous) | listing S3 | free sample "for testing" | Binance BTC-USDT `book` và `book_delta_v2` | `book` 3 ngày (2022-10-01..03), `book_delta_v2` 3 ngày (2024-04-01..03) | **Loại** (< 30 ngày) |
| 20 | Zenodo `10.5281/zenodo.20046390` | `btcusdt_lob_oct2023.tar.gz`, 308.618.431 B, md5 `58507a0f…` | CC-BY-4.0 / CC-BY-NC-4.0, nghiên cứu phi thương mại | Binance Spot REST `/api/v3/depth`, 100 level, 5 s/snapshot | **21 ngày liên tục** 2023-10-01 → 21 | **Loại**: 21 < 30 ngày ⇒ 0 fold với FIT 21/gap 6/VAL 3 |
| 21 | Zenodo `10.5281/zenodo.10600374` | `order-book-data.zip` md5 `f1c7535f…` | CC-BY-4.0 | 6 sàn, Binance BTC-USDT 2022: snapshot **mỗi giờ** (744/tháng), giá float32 | 12 tháng | **Loại** (nhịp 60 phút, mất độ chính xác giá) |
| 22 | Zenodo 8349603 / 11048480 / 21617204; figshare; Harvard Dataverse | — | — | DAX stocks, code, Bybit features; không có kết quả phù hợp | — | **Loại** |

## Kết luận

Không có nguồn miễn phí/đã có quyền nào đáp ứng đồng thời Binance Spot L2, ≥ 30 ngày lịch và segment liên tục
≥ 25,6 h trong FIT/VAL. Hai ứng viên gần nhất:

- `predict-quant/binance-spot-orderbook`: đúng Spot diff 100 ms nhưng collector đứt 20–50 lần/ngày; run dài nhất
  22,64 h < 25,6 h, VAL fold lịch chỉ ≤ 7,5 h; snapshot không có timestamp và chỉ 20 level; không có license.
- Zenodo 20046390: đúng Spot, 100 level, 5 s, liên tục nhưng chỉ 21 ngày < 30 ngày cho một fold.

Không chọn nguồn nào, không download/prepare/train thêm. Quyết định tối thiểu cần user: (a) cấp nguồn có trả phí/khóa
(Tardis API key, Binance historical data access, Crypto Lake/Kaiko…) cho ≥ 30–58 ngày Spot liên tục; hoặc
(b) chấp nhận đổi phương pháp tường minh (vd. FIT ngắn hơn cho bộ 21 ngày Zenodo, hoặc context TimesFM/h180 khác cho
predict-quant); hoặc (c) thu thập mới ≥ 30/58 ngày bằng collector liên tục.
