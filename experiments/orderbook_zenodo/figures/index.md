# Figures — TimesFM và AutoTS (hậu kỳ, chỉ từ artifact đã lưu)

Sinh bởi `python -m src_OB visualize --config <config>` từ `experiments/orderbook_zenodo`. Không train, không inference, không GPU:
chỉ đọc `predictions.parquet` của cell completed, `summary/per_fold_per_horizon.csv` và timeline mid thô đã prepare.

- Model: TimesFM zero-shot (train code 2a5a1c3), TimesFM LoRA (train code 2a5a1c3), AutoTS v2 (train code 37aae35, 6108a1d).
- Origin: ở mỗi ngày UTC nằm trong VAL, lấy origin chung đầu tiên trong các giờ 04:00, 12:00, 20:00 UTC
  (quy tắc cố định, không chọn theo lỗi hay dự báo). Mọi family và horizon chấm cùng origin, cùng actual/E0 (đã kiểm).
- Mỗi ảnh path: trục x = t → t+1 → t+2 → t+3 (h = 60 s, 120 s, 180 s); trục y = thay đổi giá so với
  mid tại t. Actual đen (mid as-of t+h, đúng giá dùng để chấm metric), E0 xám nét đứt, đường xám nhạt là mid thô
  giữa các mốc; mỗi model một màu/marker cố định, nhãn ghi |lỗi| tuyệt đối (USDT) theo từng horizon.
- Heatmap: `heatmap_fold_horizon.png` (5 fold × 3 h, từ summary) và `heatmap_day_horizon.png`
  (10 ngày VAL × 3 h, tính lại từ predictions, dữ liệu trong `day_gains.csv`). Thang màu
  diverging chung, cắt ở ±0.1; số trong ô là giá trị thật.

## Ảnh path (30)

| file | fold | origin (UTC) | mid tại t | Δ actual t+1 / t+2 / t+3 (USDT) |
|---|---|---|---|---|
| `paths/path_fold1_20231012_040130.png` | fold1 | 2023-10-12T04:01:30 | 26,843.29 | -0.57 / -6.35 / -7.88 |
| `paths/path_fold1_20231012_120006.png` | fold1 | 2023-10-12T12:00:06 | 26,839.75 | +13.23 / +15.24 / -10.67 |
| `paths/path_fold1_20231012_200007.png` | fold1 | 2023-10-12T20:00:07 | 26,711.60 | +0.06 / +0.06 / +5.05 |
| `paths/path_fold1_20231013_040028.png` | fold1 | 2023-10-13T04:00:28 | 26,783.96 | -2.16 / -2.16 / -5.82 |
| `paths/path_fold1_20231013_120001.png` | fold1 | 2023-10-13T12:00:01 | 26,823.79 | +0.32 / -5.70 / -11.71 |
| `paths/path_fold1_20231013_200005.png` | fold1 | 2023-10-13T20:00:05 | 26,773.03 | +5.46 / +8.20 / +11.97 |
| `paths/path_fold2_20231014_040002.png` | fold2 | 2023-10-14T04:00:02 | 26,931.85 | +0.00 / -3.85 / -2.65 |
| `paths/path_fold2_20231014_120113.png` | fold2 | 2023-10-14T12:01:13 | 26,874.57 | +1.76 / -2.20 / -6.33 |
| `paths/path_fold2_20231014_200005.png` | fold2 | 2023-10-14T20:00:05 | 26,862.32 | -3.51 / -3.51 / -3.51 |
| `paths/path_fold2_20231015_040023.png` | fold2 | 2023-10-15T04:00:23 | 26,881.85 | +2.88 / +7.36 / +7.36 |
| `paths/path_fold2_20231015_120000.png` | fold2 | 2023-10-15T12:00:00 | 26,851.33 | -4.16 / -9.80 / -9.80 |
| `paths/path_fold2_20231015_200002.png` | fold2 | 2023-10-15T20:00:02 | 27,046.75 | -16.40 / -5.95 / -8.15 |
| `paths/path_fold3_20231016_040005.png` | fold3 | 2023-10-16T04:00:05 | 27,238.35 | +1.64 / +5.69 / +5.69 |
| `paths/path_fold3_20231016_120006.png` | fold3 | 2023-10-16T12:00:06 | 27,751.67 | -7.27 / -10.38 / -16.69 |
| `paths/path_fold3_20231016_200000.png` | fold3 | 2023-10-16T20:00:00 | 28,468.85 | +17.82 / -19.95 / -29.54 |
| `paths/path_fold3_20231017_040003.png` | fold3 | 2023-10-17T04:00:03 | 28,268.53 | -11.88 / -104.54 / -79.21 |
| `paths/path_fold3_20231017_120014.png` | fold3 | 2023-10-17T12:00:14 | 28,425.88 | +3.11 / +17.03 / +21.25 |
| `paths/path_fold3_20231017_200001.png` | fold3 | 2023-10-17T20:00:01 | 28,511.81 | -3.86 / +4.19 / +17.48 |
| `paths/path_fold4_20231018_040011.png` | fold4 | 2023-10-18T04:00:11 | 28,520.25 | -15.25 / -30.25 / -19.72 |
| `paths/path_fold4_20231018_120000.png` | fold4 | 2023-10-18T12:00:00 | 28,340.25 | +18.23 / +23.20 / +29.76 |
| `paths/path_fold4_20231018_200001.png` | fold4 | 2023-10-18T20:00:01 | 28,251.36 | -27.29 / -6.84 / +13.57 |
| `paths/path_fold4_20231019_040040.png` | fold4 | 2023-10-19T04:00:40 | 28,256.18 | -3.76 / +1.53 / -1.35 |
| `paths/path_fold4_20231019_120000.png` | fold4 | 2023-10-19T12:00:00 | 28,450.97 | +10.99 / +19.02 / +19.04 |
| `paths/path_fold4_20231019_200001.png` | fold4 | 2023-10-19T20:00:01 | 28,753.62 | -13.57 / -0.87 / +4.08 |
| `paths/path_fold5_20231020_040001.png` | fold5 | 2023-10-20T04:00:01 | 29,240.00 | +0.36 / +30.32 / +40.11 |
| `paths/path_fold5_20231020_120002.png` | fold5 | 2023-10-20T12:00:02 | 29,843.17 | +19.12 / +23.13 / +36.82 |
| `paths/path_fold5_20231020_200000.png` | fold5 | 2023-10-20T20:00:00 | 29,560.65 | -30.39 / -42.34 / -25.40 |
| `paths/path_fold5_20231021_040119.png` | fold5 | 2023-10-21T04:01:19 | 29,605.29 | +0.70 / +0.70 / +0.70 |
| `paths/path_fold5_20231021_120002.png` | fold5 | 2023-10-21T12:00:02 | 29,808.50 | +8.24 / +5.54 / -9.39 |
| `paths/path_fold5_20231021_200001.png` | fold5 | 2023-10-21T20:00:01 | 30,148.76 | -3.58 / -11.48 / -23.76 |
