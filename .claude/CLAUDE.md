# P0_forecasting — Order Book / branch OB

Luồng đang dùng: `src_OB/`, `configs/orderbook.json`, `src_OB/README.md` và
`docs/VAST_SESSION_PROMPT.md`. `src/p0/` là code cũ được giữ lại, chỉ import helper cần thiết.
`docs/RESEARCH_PLAN.md`, các report đề xuất cũ và `docs/archive/` không điều khiển run OB.
Khi mâu thuẫn, yêu cầu mới nhất của user thắng. Không tự thêm model, feature search hoặc quy trình nghiên cứu.

## Cách vận hành

- Python CLI tự lặp fold/model/horizon. Session chính chạy `download → prepare → train → summarize`,
  xử lý lỗi thực tế và theo dõi process trong cùng một goal; không cần agent điều phối từng model.
- Chỉ giữ agent `checker`: đọc code, dữ liệu/metadata đã sinh và artifact để báo lỗi. Không train, không sửa code,
  không tự gọi agent khác. Session chính sửa lỗi và tiếp tục phần việc đã được user cho phép.
- User đã cấm **smoke test, canary, unit/integration test, synthetic run, trial fit, benchmark pass**.
  Không chạy chúng dưới tên preflight/check/probe khác. Được đọc code/log/metadata, xem thiết bị và xem kết quả
  của download/prepare/training thật. Guard sequence/causality/GPU trong đường chạy thật vẫn bắt buộc.
- Không dùng `scripts/vast_bootstrap.sh`, các `*canary*`, `gpu-probe` hoặc workflow `run.py` cũ cho OB.
- Code đã viết nhưng chưa được xác nhận bằng full run. Chỉ ghi PASS/hoàn tất khi có bằng chứng thực tế.

## Data và phương pháp cố định

- BTCUSDT Binance Spot; HF `MaximumLeverage/crypto-lob-stream`, pin full commit SHA theo config.
  Public archive không phải 2 năm. Coverage và số fold lấy từ dữ liệu reconstruct hợp lệ.
- Depth row là diff price-level. Gom trọn message, replay từ snapshot theo U/u; quantity là giá trị mới tuyệt đối,
  0 thì xóa; prune cache sau mỗi message rồi lấy top 10 bid + top 10 ask.
- Gap sequence/timestamp, book không hợp lệ và hard gap 2026-07-05 20:56–21:39 UTC: kết thúc segment,
  chờ snapshot mới. Không nối/forward-fill qua gap, không che lỗi collector June–August 2026.
- Một mid-price `(bestBid + bestAsk)/2`. OF theo price/quantity giữa state liên tiếp; OFI = bidOF − askOF.
  Tính và cộng flow trước khi drop same-mid. Raw timestamp/mid timeline được giữ riêng trước khi lọc.
- Direct: một model/adapter mỗi fold/horizon; h = 60/120/180 giây. Label và context as-of `timestamp <= query`,
  trong cùng segment. Gap train/VAL theo config: HF 6 ngày; Zenodo 21 ngày dùng FIT 9d/gap 1d/VAL 2d/step 2d
  (user quyết định 2026-09-11; gap phải > horizon dài nhất). Scaler/fit/search chỉ dùng FIT, không chọn tham số bằng outer VAL.
- Nguồn đang chạy (user chọn 2026-09-11 sau khi HF bị chặn): Zenodo 20046390, `configs/orderbook_zenodo.json`,
  full snapshot REST ~1,24 s (`src_OB/snapshots.py`), OF là flow giữa snapshot liên tiếp; nghiên cứu phi thương mại.
- Baseline chỉ OF/OFI + timing, không distance, không DeepLOB-inspired. Các family: lgbm, xgb, cat, xgbrf,
  lstm, autots, tfm_zero_shot, tfm_lora. AutoTS search trong GPU allowlist; không feature subset search.
  TimesFM zero-shot chỉ chuỗi mid-price; LoRA + OF head cùng optimizer, không XReg.
- RMSE/MAE/R² trên raw price. `rmse_gain_vs_e0` và `r2_os_vs_e0` dùng E0 hiện có cùng fold/horizon/origins.
  Latency đo trong inference thật; p95/p99 batch 1 và max quan sát được, không thêm pass benchmark.

## Authorization và xử lý lỗi

- Prompt Vast do user gửi là authorization cho goal OB trên **máy Vast đã được cấp**: chuẩn bị data,
  sửa lỗi data/env/tích hợp trong scope và training tất cả model. Không yêu cầu unlock lại giữa các bước.
  Việc sửa tài liệu ở local không tự khởi chạy training.
- Fit model **GPU-only**, không fallback CPU. CPU dùng cho I/O/reconstruct/features/scaler/metric và inference
  vốn chạy CPU của LightGBM/CatBoost. Thiếu GPU hoặc backend lỗi: dừng fit, sửa env/thiết bị trong phạm vi đã cấp;
  không thuê/đổi/xóa instance, không tự hạ batch/epoch/context hoặc bỏ model để báo thành công.
- Lỗi triển khai/API/env có thể sửa mà giữ phương pháp: sửa rồi tiếp tục lượt chạy thật, lưu traceback và thay đổi.
  Không chạy smoke/test để chứng minh bản sửa. Không lặp lại nguyên lệnh lỗi mà không xử lý nguyên nhân.
- Dữ liệu hợp lệ nhưng không đủ segment/context/fold: không tạo dữ liệu giả, không giảm gap hoặc đổi methodology
  ngầm. Ghi bằng chứng và nêu đúng quyết định còn cần từ user. Goal chưa hoàn tất khi còn model bắt buộc chưa chạy.

## Artifact và lịch sử

- Không xóa/sửa raw archive đã tải để làm nó có vẻ liên tục. Tạo prepared version mới khi sửa replay.
  Hai CSV OHLCV canonical, checksum cũ, `experiments/15d/` và `Baseline_LGBM.py` giữ nguyên.
- Không ignore metric/prediction/checkpoint/log/figure. Không ghi đè cell đã hoàn tất. Recovery: `train --resume`
  (thêm 2026-09-11) bỏ qua cell completed đúng config/revision (khác thì dừng), chuyển attempt dở dang kèm
  run/failed sang `<output>/attempts/<fold>/<model>/<h>/attemptN`; cell mới ghi `prior_attempts` và `train_code`.
- Commit/push phần việc OB và artifact được phép, stage theo scope; không dùng `git add -A` kéo thay đổi ngoài run.
  Không force push, reset --hard, xóa kết quả hoặc commit secret/checkpoint pretrained gốc.
- `.claude/MEMORY.md` ghi trạng thái thật, không kế thừa PASS từ vòng OHLCV. Context compact không kết thúc goal;
  lưu run/config/stage/process/việc còn thiếu trước khi compact rồi tiếp tục.

@MEMORY.md
