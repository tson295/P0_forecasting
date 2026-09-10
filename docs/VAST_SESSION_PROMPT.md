# Prompt Claude trên Vast — một goal Order Book

Copy toàn bộ khối dưới vào session Claude đang ở repo branch `OB` trên **Vast GPU đã được cấp**.
Prompt này cho phép chạy data preparation và training thật; không cần một lần unlock riêng.
Đây không phải khẳng định code đã chạy thành công: hiện raw đã tải ở máy local, còn prepare/full run chưa chạy.

```text
Thực hiện MỘT GOAL duy nhất cho P0_forecasting trên Vast:
Hoàn thiện historical Order Book data của BTCUSDT Binance Spot từ archive HF pinned,
chạy đầy đủ pipeline src_OB cho tất cả model/fold/horizon thực tế, xuất metrics/latency/summary,
kiểm tra artifact và báo cáo kết quả. Không kết thúc goal ở bước viết plan, data xong, hay một model vừa xong.

Đây là authorization của user cho toàn bộ công việc trên instance Vast đang được cấp:
cài/build dependency GPU, tải archive, prepare/reconstruct data thật, sửa lỗi triển khai/env/adapter
trong phạm vi phương pháp đã chốt, training thật, ghi artifact, commit/push branch OB.
Không hỏi unlock/approval giữa các bước. Không tạo/thuê/xóa instance hoặc tự đổi GPU ngoài quyền đã cấp.
Không training trên máy local. Không CPU training/fallback.

Đọc theo thứ tự:
1. .claude/CLAUDE.md và .claude/MEMORY.md
2. .claude/AGENT.md và .claude/agents/checker.md
3. src_OB/README.md và configs/orderbook.json
4. Code src_OB liên quan tới stage đang xử lý.
docs/RESEARCH_PLAN.md, workflow OHLCV run.py và docs/archive là lịch sử, không điều khiển goal này.

VẬN HÀNH
- Python CLI tự lặp các family/fold/horizon; không dựng agent runner/controller/monitor/infra/researcher/analyst.
  Session chính chạy lệnh, theo dõi process, sửa lỗi và tiếp tục. Chỉ gọi checker để đọc evidence sau prepare,
  khi có lỗi correctness cụ thể, và khi tổng kết; không gọi checker trước/sau từng model.
- Cấm mọi smoke/canary/unit/integration test, synthetic run, trial fit, GPU-probe fit, benchmark pass, warmup riêng.
  Không dùng scripts/vast_bootstrap.sh, các *canary*, run.py gpu-probe, pytest hoặc các workflow cũ.
  Được đọc code/data/metadata/log, xem nvidia-smi/process và đối chiếu artifact của lượt chạy thật.
- Không suy ra code đã hoàn thiện chỉ từ commit. Nếu lượt chạy thật lộ bug/API mismatch, tự sửa trong src_OB
  và tiếp tục mà giữ các invariants dưới đây. Giữ src/p0, Baseline_LGBM.py và kết quả cũ nguyên trạng.

DATA / PHƯƠNG PHÁP KHÔNG ĐỔI
- HF MaximumLeverage/crypto-lob-stream, full SHA theo configs/orderbook.json:
  873f31e729ae23b1c309cd5dcb33feed27c407de. Chỉ BTCUSDT, exchange binance (Spot).
- Archive hiện có 4 file depth/snapshots tháng 6 và 7/2026, 704.186.850 byte. Không giả định 2 năm.
  Raw đã tải ở máy local; chỉ manifest được commit, clone trên Vast chưa chắc có các Parquet.
  Dùng downloader để lấy đúng revision, không cần Tardis key. Không đổi revision, bịa/pad lịch sử.
- Depth là diff, không phải snapshot từng row. Gom đủ một message theo timestamp/U/u; replay từ snapshot
  quan sát được trước event; quantity mới là absolute, 0 xóa; prune cache sau message rồi lấy top 10 mỗi phía.
- Gap ID/timestamp hoặc book không hợp lệ: kết thúc segment, không replay xuyên gap, chờ snapshot mới.
  2026-07-05 20:56–21:39 UTC là hard reset. Không forward-fill qua phần thiếu do collector cũ.
- Một mid; OF tính trước same-mid drop và cộng dồn tới mid-change; OFI = bidOF - askOF, 10 level.
  Baseline chỉ OF/OFI + timing. Giữ raw timestamp/mid timeline riêng để as-of sampling.
- Direct h60/120/180 giây, một model/adapter mỗi fold/horizon. Label log(MP(t+h)/MP(t)), quote cuối <= t+h.
  Feature/window/label nằm trong segment hợp lệ. Gap FIT/VAL >5 ngày, default 6 ngày.
- Family đủ: lgbm,xgb,cat,xgbrf,lstm,autots,tfm_zero_shot,tfm_lora. Không thêm DeepLOB, không feature search.
  AutoTS tự search model/tham số trong GPU allowlist. TimesFM zero-shot chỉ mid; LoRA + OF head, không XReg.
- RMSE/MAE/R² trên raw price; thêm rmse_gain_vs_e0 và r2_os_vs_e0 từ cùng E0 hiện có/cùng origins.
  Không tính metric trên log-return. Latency trong inference thật: p95/p99 batch 1 và max quan sát, không hard bound.

THỰC HIỆN LIÊN TỤC
1. Ở checkout OB, ghi commit code, config, GPU/driver/package versions của môi trường thật vào thư mục run.
   Chuẩn bị CUDA-enabled PyTorch, LightGBM build gpu/cuda phù hợp image và requirements-vast.txt.
   Requirements hiện hướng tới CUDA 12. Không cài JAX/XReg hoặc gọi bootstrap cũ. Không fit thử để chọn backend;
   sử dụng backend GPU tường minh, xử lý lỗi nếu actual fit thất bại. Một process nặng/GPU, không cộng VRAM nhiều GPU.

2. Chạy download thật:
   python -m src_OB download --config configs/orderbook.json
   Downloader tiếp tục file đã hoàn tất; manifest không thay thế cho file Parquet thực tế trên Vast.

3. Hoàn thiện data bằng prepare thật:
   python -m src_OB prepare --config configs/orderbook.json
   Nếu prepared_dir đã có: đọc manifest/status để phân biệt complete/partial/revision khác. Không xóa để chạy đè;
   reuse chỉ khi cùng contract, nếu sửa replay thì tạo prepared version mới và lưu nguyên nhân/config.
   Ghi DATA_REPORT.md trong thư mục output: coverage thực, n raw/kept states, segment duration distribution,
   gap/reset reasons, known gap handling, số fold và eligible train/VAL origins theo đúng mask của pipeline.
   Đây là kết quả xử lý dữ liệu thật, không synthetic test hoặc model benchmark.

4. Checker đọc data/metadata/code trước training. Chú ý: context TimesFM 512 điểm cách nhau h giây;
   ở h180 cần 25h33 context liên tục, chưa tính label. Snapshot reanchor/missing updates có thể làm common
   origin mask rỗng cho tất cả model. Đọc bằng chứng thật, sửa lỗi replay nếu có; không nối gap, bỏ model,
   giảm context/gap ngầm hoặc khẳng định data khả dụng chỉ vì đủ tháng trên lịch.
   Số fold có thể ít hơn 5 theo coverage như code đã cho phép. Nếu phương pháp hiện tại không thể chạy trên
   các segment hợp lệ, nêu blocker có số liệu và quyết định data/context còn cần; không báo goal hoàn tất giả.

5. Nếu data khả dụng, chạy FULL training thật ngay, không dừng chờ user duyệt:
   export P0_OB_VAST=1
   export CUDA_VISIBLE_DEVICES=0
   python -m src_OB train --config configs/orderbook.json
   python -m src_OB summarize --config configs/orderbook.json
   Chỉ chạy lệnh sau khi lệnh trước thành công. CLI tự xử lý toàn bộ model/fold/horizon.
   Giữ process/log trong tmux hoặc cơ chế chạy bền của Vast để SSH/context đổi không làm mất run.
   Nếu dùng nhiều GPU, chia các fold thật thành tập rời nhau bằng --folds, cùng config và common mask;
   không mở nhiều agent điều phối và không chạy trùng cell.

6. Nếu lỗi runtime/env/GPU build/API: giữ traceback/artifact, sửa nguyên nhân rồi tiếp tục công việc thật.
   Không fallback CPU, không hạ batch/epoch/context, đổi seed/target/split hoặc giảm search để lách lỗi.
   Không lặp nguyên lệnh lỗi vô hạn. Không bỏ model lỗi rồi tổng kết như đầy đủ.
   CLI hiện CHƯA có --resume/--horizons, completed directory không được overwrite. Nếu run bị ngắt,
   session chính được bổ sung recovery theo cell: reuse completed chỉ khi khớp code/config/data phù hợp,
   lưu failed attempt và chạy lại cell chưa hoàn tất, không giả nhận checkpoint thiếu optimizer là resume chính xác.
   Không train lại cell thành công chỉ để có thêm số đo latency. Nếu correctness fix làm artifact cũ mất hiệu lực,
   lưu riêng chúng và ghi rõ các cell cần chạy lại; không trộn config/revision khác trong summary.

7. Sau khi hết cell, checker đọc artifact và summary, không infer/fit lại.
   Expected cells = số fold THỰC TẾ × 8 family × 3 horizon. Mỗi cell phải có completed.json,
   metrics.json/metrics.csv, predictions.parquet, latency.json/inference_latency.csv và checkpoint/adapter
   phù hợp (zero-shot chỉ cần pretrained ID/revision). Không chỉ đếm thư mục hoặc nhìn exit code.
   Summary phải có per_fold_per_horizon.csv và by_model_horizon.csv, gồm metrics giá, E0 gains và latency.
   E0 denominator=0 được để null có lý do, không ép số để checker PASS.

8. Viết RUN_REPORT.md: coverage/segment/cell counts, kết quả theo model/horizon, hai E0 gains, latency,
   runtime thật, GPU/env/code/config provenance, lỗi đã sửa, phần chưa đạt và giới hạn dữ liệu.
   Lưu checker findings cùng report. Cập nhật MEMORY theo trạng thái thực; không dùng PASS của OHLCV cũ.
   Commit/push code, config, reports và artifact OB theo scope trên branch OB; LFS cho file lớn đã cấu hình.
   Không git add -A, force push, xóa dữ liệu/kết quả, commit secret hoặc checkpoint pretrained gốc.

GIỮ MỘT GOAL
Không dừng vì data vừa xong, một model xong, cần commit, thời gian dài hoặc cần compact context.
Cập nhật trạng thái định kỳ từ process/log thật. Lưu exact next step/run/config trước compact rồi tiếp tục goal.
Chỉ báo COMPLETE khi data hợp lệ và tất cả cell bắt buộc cùng artifact/summary/report đã hoàn tất.
Nếu bị chặn thật bởi dữ liệu không khả dụng, quyền truy cập hoặc GPU/tài nguyên không thể sửa trong quyền hiện có,
ghi rõ bằng chứng, công việc đã xong, cell còn thiếu và thông tin/quyết định cần từ user; không gọi đó là hoàn tất.
Bắt đầu thực hiện trên Vast ngay sau khi đọc context; không chỉ trả lời bằng kế hoạch.
```
