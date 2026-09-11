# P0_forecasting — branch tfm_autots

Phase hiện tại trên branch này: chỉ TimesFM và AutoTS của pipeline OHLCV `src/p0`.
Đọc `docs/TFM_AUTOTS_PHASE.md`, `configs/tfm_autots.json` và `src/p0/phase_tfm_autots.py`.
Lệnh chạy thật trên Vast: `P0_TFM_AUTOTS_VAST=1 python run.py tfm-autots`.
User hiện chỉ yêu cầu code/commit/push; không tự chạy training ở phiên sửa code.
Cấm mọi test/smoke/canary/probe fit/benchmark/warmup riêng. Chưa có bằng chứng runtime cho code mới.
Session chính làm việc, không gọi thêm agent nếu user chưa yêu cầu. Các ghi chú OB dưới đây là context
của nhánh trước, không điều khiển phase tfm_autots và không yêu cầu chạy các model OB khác.

## Context Order Book được giữ lại

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
  trong cùng segment. Gap train/VAL > 5 ngày. Scaler/fit/search chỉ dùng FIT, không chọn tham số bằng outer VAL.
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
- Không ignore metric/prediction/checkpoint/log/figure. Không ghi đè cell đã hoàn tất; CLI hiện chưa có resume
  hoàn chỉnh theo cell. Khi cần recovery, session chính bổ sung skip/select cell đúng config/revision hoặc lưu
  attempt mới có provenance, không tuyên bố CLI đã có `--resume` khi chưa thêm.
- Commit/push phần việc OB và artifact được phép, stage theo scope; không dùng `git add -A` kéo thay đổi ngoài run.
  Không force push, reset --hard, xóa kết quả hoặc commit secret/checkpoint pretrained gốc.
- `.claude/MEMORY.md` ghi trạng thái thật, không kế thừa PASS từ vòng OHLCV. Context compact không kết thúc goal;
  lưu run/config/stage/process/việc còn thiếu trước khi compact rồi tiếp tục.

@MEMORY.md
