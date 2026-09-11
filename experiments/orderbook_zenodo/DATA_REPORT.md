# DATA_REPORT — BTCUSDT Binance Spot L2

Sinh bởi `python -m src_OB data-report` từ source đã tải và prepared dataset thật (memmap/metadata của `prepare`; mask origin dùng chung hàm `select_origins` với `train`). Không fit/infer model, không dữ liệu tổng hợp.

**Trạng thái: READY**

## 1. Nguồn

- `zenodo:20046390` revision `58507a0fe055f0f2d3a498c15fc7d8d2`; prepared schema v3 tại `data/orderbook/prepared_zenodo_20046390`.
- Replay 2; code `c68fc452378172cba9f10af04d85d1f39fce9eb1`; config_sha256 `78aa19aa43d3560403d43394e8003fae6da3eeaf61a4de35eabcd9e0c4890ac7`.
- Nguồn: Zenodo 20046390 ccxt REST full top-100 snapshots (~1.24 s cadence) (full_snapshots).
- File: `btcusdt_lob_oct2023.tar.gz`.

## 2. Snapshot đầy đủ sau khi nạp

- 1,439,157 snapshot hợp lệ, từ 2023-10-01T00:00:04.385000+00:00 tới 2023-10-21T23:59:59.893000+00:00 (span 503.999 h).
- Bước thời gian giữa hai snapshot (giây): {'q0': 1.222, 'q0.01': 1.225, 'q0.05': 1.227, 'q0.25': 1.234, 'q0.5': 1.236, 'q0.75': 1.245, 'q0.95': 1.283, 'q0.99': 1.691, 'q1': 6.687}; số bước > 10 s: 0.
- Mỗi snapshot là book đầy đủ (top 100 mỗi phía): không replay diff; OF là flow quan sát giữa hai snapshot liên tiếp, không phải flow từng message.

## 3. Dựng book thật (`prepare`)

- Raw state: **1,439,157**; origin giữ lại sau same-mid drop: **237,648** (mỗi segment tính cả state đầu tiên).
- Coverage theo segment: 2023-10-01T00:00:04.385000+00:00 → 2023-10-21T23:59:59.893000+00:00; tổng thời gian segment hợp lệ **1814394.242 s**; segment dài nhất 1814392.989 s.
- Đếm: {'snapshot_messages': 1439159, 'invalid_snapshot_line': 1, 'timestamp_not_increasing': 1, 'snapshots_dropped_out_of_order': 1}.
- Lý do kết thúc segment: {'timestamp_not_increasing': 1, 'archive_end': 1}; reset: {'timestamp_not_increasing': 1}.
- Hard gap khai báo: không có.
- Segment: 2; thời lượng (s) {'q0': 1.253, 'q0.01': 18145.17, 'q0.05': 90720.84, 'q0.25': 453599.187, 'q0.5': 907197.121, 'q0.75': 1360795.055, 'q0.95': 1723673.402, 'q0.99': 1796249.072, 'q1': 1814392.989}; state/segment {'q0': 2.0, 'q0.01': 14393.53, 'q0.05': 71959.65, 'q0.25': 359790.25, 'q0.5': 719578.5, 'q0.75': 1079366.75, 'q0.95': 1367197.35, 'q0.99': 1424763.47, 'q1': 1439155.0}.

| id | start_utc | last_utc | seconds | states | kept_origins | end_reason |
|---|---|---|---|---|---|---|
| 0 | 2023-10-01T00:00:04.385000+00:00 | 2023-10-01T00:00:05.638000+00:00 | 1.253 | 2 | 1 | timestamp_not_increasing |
| 1 | 2023-10-01T00:00:06.904000+00:00 | 2023-10-21T23:59:59.893000+00:00 | 1814392.989 | 1439155 | 237647 | archive_end |

## 4. Walk-forward và origin theo mask của pipeline

Fold lấy từ coverage theo lịch (train 9 d, gap 1 d, VAL 2 d, bước 2 d, tối đa 5). Số đếm là origin còn lại sau từng mask, cộng dồn qua các horizon (tập chung cho mọi family).

| fold | train_start | train_end | val_start | val_end | valid_segment_seconds_fit | valid_segment_seconds_val | train_label_context | train_context_inside_fit | val_label_context | train_price_context_h60s | val_price_context_h60s | train_price_context_h120s | val_price_context_h120s | train_price_context_h180s | val_price_context_h180s | train_origins | val_origins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fold1 | 2023-10-01T23:59:59.893001+00:00 | 2023-10-10T23:59:59.893001+00:00 | 2023-10-11T23:59:59.893001+00:00 | 2023-10-13T23:59:59.893001+00:00 | 777600.0 | 172800.0 | 104663 | 104564 | 17447 | 98704 | 17447 | 90772 | 17447 | 83671 | 17447 | 83671 | 17447 |
| fold2 | 2023-10-03T23:59:59.893001+00:00 | 2023-10-12T23:59:59.893001+00:00 | 2023-10-13T23:59:59.893001+00:00 | 2023-10-15T23:59:59.893001+00:00 | 777600.0 | 172800.0 | 93248 | 93149 | 7621 | 89690 | 7621 | 85187 | 7621 | 80438 | 7621 | 80438 | 7621 |
| fold3 | 2023-10-05T23:59:59.893001+00:00 | 2023-10-14T23:59:59.893001+00:00 | 2023-10-15T23:59:59.893001+00:00 | 2023-10-17T23:59:59.893001+00:00 | 777600.0 | 172800.0 | 80942 | 80843 | 31161 | 78993 | 31161 | 71318 | 31161 | 66173 | 31161 | 66173 | 31161 |
| fold4 | 2023-10-07T23:59:59.893001+00:00 | 2023-10-16T23:59:59.893001+00:00 | 2023-10-17T23:59:59.893001+00:00 | 2023-10-19T23:59:59.893001+00:00 | 777600.0 | 172800.0 | 83589 | 83490 | 25130 | 81189 | 25130 | 77713 | 25130 | 75100 | 25130 | 75100 | 25130 |
| fold5 | 2023-10-09T23:59:59.893001+00:00 | 2023-10-18T23:59:59.893001+00:00 | 2023-10-19T23:59:59.893001+00:00 | 2023-10-21T23:59:59.893001+00:00 | 777600.0 | 172800.0 | 89839 | 89740 | 31388 | 88062 | 31388 | 82877 | 31388 | 79968 | 31388 | 79968 | 31388 |

## 5. Context cần so với segment thực tế

| horizon_seconds | price_context_points | price_context_span_hours | segment_hours_needed_with_label |
|---|---|---|---|
| 60 | 512 | 8.517 | 8.533 |
| 120 | 512 | 17.033 | 17.067 |
| 180 | 512 | 25.55 | 25.6 |

LSTM/tree cần 100 origin mid-change liên tiếp cùng segment và nhãn tại t+h trong segment; TimesFM/AutoTS cần thêm context giá cách đều h như bảng. Segment hợp lệ dài nhất: 1814392.989 s; bước snapshot lớn nhất: 6.687 s.

