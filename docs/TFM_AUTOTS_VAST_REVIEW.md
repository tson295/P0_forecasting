# Rà soát trước Vast — 2026-09-11

Phạm vi: đọc code/manifest và sửa implementation trên `tfm_autots`. Không chạy test, smoke, probe,
training, inference hoặc benchmark. Chưa thể chứng nhận hết lỗi runtime hoặc đưa ra thời gian hoàn thành.

## Data đang dùng

Config `configs/tfm_autots.json` trỏ tới `data/BTC_1m_2y.csv`. Manifest đã lưu ghi 1.051.201 bar,
2024-09-03 16:29:00 → 2026-09-03 16:29:00 UTC, 730 ngày. Đây là OHLCV BTC 1m, không phải Order Book.
`BTC_5m_2y.csv` là dữ liệu dẫn xuất phục vụ features 5m. Git HEAD chứa LFS pointers của cả hai file,
có OID khớp SHA trong `data/data_checksums_2y.json`. Chưa tải lại/đọc hết CSV để xác minh trong phiên này.
CLI đối chiếu checksum và data contract khi load data cho lượt chạy thật; không ghi lại checksum để lách lỗi.

## Chi phí lặp đã xử lý

| Chỗ gây chi phí | Đường thực thi hiện tại |
|---|---|
| AutoTS dựng/scaling full-grid và native training features mỗi candidate | Chuẩn bị pool theo fold trước search; candidate chọn cột và bootstrap rows rồi fit estimator GPU |
| TimesFM dựng/scaling full-grid covariates mỗi candidate | Dùng cùng cơ chế pool theo fold, tại `series_preprocess/tfm`; không encode z-target vốn không được dùng |
| TimesFM forecast lặp theo candidate/covariate permutation | Forecast batch, cache theo adapter/data/origins; residual heads dùng features + forecast |
| LoRA gọi `.item()` cho loss ở mỗi batch | Cộng loss trên device bằng float64, chỉ đọc scalar cuối epoch/ES; giữ loss/backprop, seed, batch và optimizer |
| AutoTS-final fit lại winner tại seed selection trong confirmation | Dùng lại RMSE/predictions đã tính đúng cùng seed, fold, template và feature set |
| LoRA im lặng khi chạy lâu | Ghi số windows/batch/epoch và tiến độ mỗi epoch; phase ghi thời gian từng stage |
| SSH reconnect vô tình khởi chạy thêm full phase | Launcher giữ `flock`, lưu log riêng từng lần và chọn `--resume` khi có progress |

## Chi phí vẫn còn theo phương pháp

- TimesFM LoRA không chỉ train một adapter cho toàn phase: adapter tách theo fold, seed và epoch mode.
  FIT rolling 120 ngày, trừ inner ES/residual calibration/purge; `train_stride=1`, context 512,
  batch train 64, tối đa 20 epochs. Flip invariance dùng hai forward branches. Cache không bỏ các fit hợp lệ này.
- AutoTS add-one tuần tự do KEEP thay đổi feature set tiếp theo. Mỗi candidate vẫn cần fit các fold.
  WR LightGBM có ba leaf estimators cho multi-output. MR bước 2–3 vẫn tính recursive features từ prediction riêng.
- Bake-off cuối giữ 4 templates và 10 internal validations: nếu hai frozen feature sets khác nhau,
  có thể tới 2 × 5 × 4 × (1 + 10) = 440 lượt đánh giá template, chưa tính outer scoring/confirmation.
  Số thực tế phụ thuộc AutoTS validation eligibility và dedup. Đây không phải vòng smoke/benchmark.
- CPU vẫn làm I/O, chọn/copy cột, metrics; LightGBM inference vốn chạy CPU. GPU-only áp dụng cho training.
- LoRA/residual regression CUDA float64, serialization model và upload LFS có chi phí riêng.
  Chưa có số đo để nói fit hay prediction là bottleneck lớn nhất trên GPU Vast được cấp.

## Vận hành và giới hạn recovery

Đọc `docs/VAST_SESSION_PROMPT.md`; goal ngắn có `/goal` ở `docs/VAST_GOAL.txt`.
Claude chính thiết lập env, chạy launcher trong tmux, theo dõi và push. Checker chỉ đọc evidence.
Đã bỏ `DISABLE_AUTO_COMPACT=1` trong settings để session dài có thể compact và tiếp tục.

`--resume` giữ stage/candidate progress và cache có cùng contract. Nó chưa resume optimizer LoRA giữa epoch
hoặc từng internal validation của AutoTS bake-off; phần chưa hoàn tất có thể phải chạy lại.
Thay code/config làm contract cũ không còn hợp lệ: không sửa hash để ép reuse. Giữ attempt cũ và ghi rõ
phạm vi artifact còn hợp lệ trước khi xử lý recovery. Không hạ epoch/context/candidates/validations để tạo tốc độ giả.
