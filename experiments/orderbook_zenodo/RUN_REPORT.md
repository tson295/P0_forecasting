# RUN_REPORT — Zenodo 20046390, BTCUSDT Binance Spot L2 (full snapshot)

**Trạng thái: RUNNING (bản nháp).** Training FULL đang chạy trên Vast; báo cáo này được hoàn thiện khi train và
summarize xong và checker cuối run đã đọc artifact. Chưa có kết luận metric tổng hợp.

## 0. Quyết định của user (2026-09-11, trong session)

- Archive HF pinned bị chặn vì dữ liệu (`experiments/orderbook_hf/RUN_REPORT.md`). Trong các nguồn đã xét
  (`experiments/orderbook_hf/SOURCE_REPORT.md`), user chọn **Zenodo 20046390**: Binance Spot BTCUSDT, snapshot REST
  top-100 mỗi phía (nhịp đo được ~1,24 s), 2023-10-01 → 10-21. OF/OFI là flow quan sát giữa hai snapshot liên tiếp,
  không phải flow từng message. License CC-BY-4.0 / CC-BY-NC-4.0, dùng cho nghiên cứu phi thương mại (user xác nhận).
- Walk-forward cho 21 ngày: **FIT 9 ngày / gap 1 ngày / VAL 2 ngày / step 2 ngày / tối đa 5 fold**. Gap chỉ cần
  lớn hơn horizon dài nhất (180 s); nhãn FIT vẫn phải kết thúc trước `train_end`.
- Model, feature, context (LSTM 100 origin, TimesFM 512), seed 8587, epoch/batch/budget AutoTS giữ như config HF.

## 1. Việc đã chạy thật

| Bước | Lệnh / commit | Kết quả |
|---|---|---|
| Download | `python -m src_OB download --config configs/orderbook_zenodo.json` | tar 308.618.431 B, md5 `58507a0f…` khớp pin |
| Prepare | `python -m src_OB prepare …` (code c68fc45) | `data/orderbook/prepared_zenodo_20046390/` replay_version 2 |
| Data report | `python -m src_OB data-report …` | **READY**, 5 fold (`DATA_REPORT.md`) |
| Checker trước train | agent `checker` | không ERROR; Z-W1 đã sửa (`CHECKER_FINDINGS.md`) |
| Train lần 1 | `run_meta/run_train.sh` (code d8797c5), 02:46:25Z | 15 cell completed; dừng ở fold1/autots/h60s (mục 3) |
| Train lần 2 | `run_train.sh --resume` (code 688fefd), 02:58:19Z | skip 15 cell; SIGSEGV ở AutoTS candidate 9 (mục 3) |
| Train lần 3 | `run_train.sh --resume` (code 2a5a1c3), 03:02:28Z | đang chạy |

Thời gian đo trong run thật (fold1): 4 family cây + LSTM khoảng 2 phút cho 15 cell; AutoTS khoảng 3 phút/cell; TimesFM
zero-shot khoảng 3 phút/cell; TimesFM LoRA khoảng 9,3 phút/epoch, tức khoảng 50 phút/cell với 5 epoch (GPU khoảng 61%,
process 100% một lõi CPU vì overhead launch khi forward/backward). Để giữ latency p95/p99 không bị nhiễu, run dùng một
process duy nhất trên GPU, không chạy fold song song trên cùng GPU; epoch/batch/context giữ nguyên.

Không chạy smoke/canary/test/probe/trial fit/benchmark/warmup riêng. Replay tham số AutoTS theo seed (mục 3) chỉ gọi
bộ sinh tham số ngẫu nhiên, không đọc data và không fit.

## 2. Data (tóm tắt từ `DATA_REPORT.md`)

- 1.439.157 snapshot hợp lệ, 2023-10-01 00:00:04 → 10-21 23:59:59 UTC; bước median 1,236 s, max 6,687 s, 0 bước > 10 s.
- 2 segment: segment 0 dài 1,25 s (đóng vì một snapshot đi lùi timestamp/nonce ở đầu ngày 10-01), segment 1 liên tục
  1.814.393 s. 237.648 origin mid-change.
- Fold (train / val origin): fold1 83.671 / 17.447, fold2 80.438 / 7.621, fold3 66.173 / 31.161,
  fold4 75.100 / 25.130, fold5 79.968 / 31.388. VAL nối tiếp 10-11 → 10-21, không chồng nhau.
- Lưu ý (checker Z-I1…I4): datacite ghi "5-second" nhưng nhịp thật ~1,24 s và timestamp là đồng hồ collector; ngày
  10-01 bất thường nhưng nằm ngoài mọi FIT/VAL/context; FIT hiệu dụng ~7,9 ngày vì context TimesFM h180 (25,55 h) phải
  nằm trong FIT; fold2 VAL ít origin (cuối tuần) nên báo cả mean theo fold lẫn pooled.

## 3. Sự cố runtime và cách sửa (giữ phương pháp)

1. **Lần 1 — LightGBM GOSS trên CUDA.** AutoTS candidate 6 (LightGBM `boosting_type=goss`) báo
   `[CUDA] invalid argument … boosting/goss.hpp 66`; `GPURegressor` đổi thành `GPUOnlyError`, không thử lại bằng CPU.
   Nguyên nhân trong LightGBM 4.7.0: `GOSSStrategy` không cấp phát `cuda_bag_data_indices_` (chỉ `bagging.hpp:167` làm
   việc này) nhưng vẫn copy vào nó.
   Khi rà cùng search space: AutoTS sample `linear_tree=True` (50%), mà `config.cpp:426-430` của LightGBM **tự chuyển
   cả lần fit sang CPU**, chỉ in cảnh báo C++ (bị `verbose=-1` che). Replay tham số theo seed cho thấy candidate 3 của
   lần 1 là `linear_tree=True`, tức **attempt 1 đã có một candidate fit trên CPU mà guard cũ không phát hiện**.
   Attempt đó không hoàn tất và không có kết quả nào được dùng; nó được giữ nguyên làm bằng chứng tại
   `attempts/fold1/autots/h60s/attempt1/`.
   Sửa (688fefd): giống XGBoost `gblinear → gbtree`, adapter AutoTS đặt `linear_tree=False` và đổi GOSS thành `gbdt`
   (luồng random theo seed không đổi). `GPURegressor` kiểm tham số trước fit, rồi đọc config hiệu lực trong model text
   sau mỗi fit LightGBM (`device_type`, `linear_tree`, `data_sample_strategy`, objective) và dừng job nếu không phải
   CUDA. Các model `lgbm` đã completed có config hiệu lực `device_type: cuda`, `linear_tree: 0`, objective `huber`
   (có kernel CUDA).
2. **Lần 2 — SIGSEGV với `max_bin` 1000.** `--resume` skip đúng 15 cell và chuyển attempt 1. AutoTS candidate 3 và 6
   (`max_bin` 255) fit CUDA xong; candidate 9 là LightGBM đầu tiên có `max_bin` 1000 (các tham số khác: `lambda_l1` 1,
   `min_data_in_leaf` 30, lr 0,01) thì process chết với SIGSEGV (exit 139), không có traceback. `max_bin > 255` đưa
   learner CUDA sang đường bin 16-bit và best-split dùng global memory, trong khi mọi fit CUDA thành công đều dùng
   đường 8-bit. `failed.json` được session ghi hậu kiểm vào cell trước khi resume chuyển thành `attempt2`.
   Sửa (2a5a1c3): allowlist giới hạn `max_bin ≤ 255`; launcher bật `PYTHONFAULTHANDLER=1` để lần crash native sau (nếu
   có) để lại Python stack trong `logs/train.log`.
3. **Recovery.** `train --resume` (thêm ở 688fefd) bỏ qua cell completed đúng config/revision (khác thì dừng) và chuyển
   attempt dở dang sang `attempts/<fold>/<model>/<h>/attemptN`; cell chạy lại ghi `train_code` và `prior_attempts`.
   Không cell completed nào bị ghi đè.

## 3b. Phát hiện khi đọc kết quả AutoTS (không phải lỗi runtime)

- fold1/autots/h60s (code 2a5a1c3) completed. Kết quả trên outer VAL: RMSE **498,9** USDT, R² −24,9,
  `rmse_gain_vs_e0` −31,96; E0 RMSE trên cùng origin là 15,14. Validation nội bộ trong FIT của model được chọn (XGBoost,
  window 5) chỉ có RMSE 7–11.
- Nguyên nhân đọc từ artifact: 100% mid của VAL fold1 thấp hơn mức thấp nhất của FIT (VAL 26.555–27.124,68 so với FIT
  min 27.175,30). Dự báo nằm trong 27.272,91–27.313,69, bias trung vị +493. Adapter hồi quy **mức giá thô** trên grid
  h giây bằng model cây (không `normalize_window`, không transformer, theo contract trong README), nên dự báo không ra
  ngoài khoảng giá đã thấy trong FIT. Đây là giới hạn của thiết kế hiện tại, không phải lỗi runtime; các family khác
  (cây Direct, LSTM, TimesFM) dự báo log-return hoặc chuỗi đã center theo giá origin.
- Không đổi thiết kế giữa run (đó là đổi phương pháp và phải ghi đè cell đã completed). Quyết định cần user ở mục 6.
- Trong search, một candidate LightGBM `dart` trên CUDA có smape 200 và RMSE ~55k, tức dự báo khoảng −28k: DART trên CUDA
  cho giá trị sai. Candidate này bị chấm điểm loại, không được chọn.

## 4. Provenance

- Prepared: replay_version 2, code_commit c68fc45, `code_uncommitted_paths` rỗng, `config_sha256` `78aa19…` (sha của
  config JSON đã resolve). `run_meta/environment.json` ghi `2995bd…` = sha trên bytes của file config (checker Z-I6).
- Train code theo cell (`run.json` → `train_code`): d8797c5 cho 15 cell của lần 1; 2a5a1c3 cho các cell từ lần 3. Thay
  đổi giữa hai commit chỉ ở adapter AutoTS/guard LightGBM/resume, không đổi feature, nhãn, fold hay model khác.
- Môi trường: RTX 3090 24 GB, driver 595.84, CUDA 12.8; torch 2.11.0+cu128, LightGBM 4.7.0 (CUDA, NCCL shared),
  XGBoost 3.4.1 (CUDA), CatBoost 1.2.10 (GPU), timesfm 2.0.2 (checkpoint `1d952420…`), AutoTS 1.0.4
  (`run_meta/environment.json`, `pip_freeze.txt`).

## 5. Kết quả

Chờ train/summarize (kỳ vọng 5 fold × 8 family × 3 horizon = 120 cell).

## 6. Quyết định còn cần từ user

1. **Target của AutoTS (mục 3b).** Adapter hiện tại hồi quy mức giá thô bằng model cây, nên dự báo luôn nằm trong khoảng
   giá của FIT. Mỗi khi giá VAL đi ra ngoài khoảng đó, sai số trở thành một độ lệch cố định (fold1: RMSE ~500 so với E0
   ~15–25). Phương án đề xuất, cần user duyệt vì làm thay đổi phương pháp AutoTS: chuỗi mà `WindowRegression` nhìn thấy
   dùng log-price đã center theo giá tại origin (giống cách TimesFM đang làm), target là `log(MP(t+h)/MP(t))`; search,
   GPU allowlist, regressor OF và fold giữ nguyên. Sau đó chạy lại 15 cell AutoTS vào một output/attempt mới có
   provenance; các cell AutoTS hiện tại được giữ lại, không ghi đè. Nếu không duyệt, kết quả AutoTS được báo cáo nguyên
   trạng cùng chẩn đoán này.
