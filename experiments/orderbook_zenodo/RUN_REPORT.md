# RUN_REPORT — Zenodo 20046390, BTCUSDT Binance Spot L2 (full snapshot)

**Trạng thái: COMPLETE.** Đủ 120/120 cell (5 fold hợp lệ × 8 family × 3 horizon), mỗi cell có completed, metrics,
predictions, latency và checkpoint (zero-shot dùng checkpoint pretrained đã pin). AutoTS là **adapter v2** (log-return
center theo origin của cửa sổ), do user duyệt; 15 cell AutoTS v1 (giá thô) được giữ ở `superseded/autots_raw_price/`.
Summary đã sinh lại; checker cuối run (ZF) và checker lượt AutoTS v2 (ZA) đều không có ERROR (`CHECKER_FINDINGS.md`, lượt ZF và ZA). Không còn quyết định mở.

## 0. Quyết định của user (2026-09-11, trong session)

- Archive HF pinned bị chặn vì dữ liệu (`experiments/orderbook_hf/RUN_REPORT.md`). Trong các nguồn đã xét
  (`experiments/orderbook_hf/SOURCE_REPORT.md`), user chọn **Zenodo 20046390**: Binance Spot BTCUSDT, snapshot REST
  top-100 mỗi phía (nhịp đo được ~1,24 s), 2023-10-01 → 10-21. OF/OFI là flow quan sát giữa hai snapshot liên tiếp,
  không phải flow từng message. License CC-BY-4.0 / CC-BY-NC-4.0, dùng cho nghiên cứu phi thương mại (user xác nhận).
- Walk-forward cho 21 ngày, thay FIT21/gap6/VAL3/step7 của config HF: **FIT 9 ngày / gap 1 ngày / VAL 2 ngày / step
  2 ngày / tối đa 5 fold**. Gap chỉ cần lớn hơn horizon dài nhất (180 s); nhãn FIT vẫn kết thúc trước `train_end`.
- Model, feature, context (LSTM 100 origin, TimesFM 512), seed 8587, epoch/batch/budget AutoTS giữ như config HF.
- **AutoTS v2 (duyệt sau khi đọc run đầu):** mỗi cửa sổ `WindowRegression` dùng X = log(P_window) − log(P_origin),
  target y = log(MP(t+h)/MP(t)), giá dự báo = mid_origin·exp(ŷ); center cố định theo từng cửa sổ; regressor OF/OFI +
  elapsed tại origin như cũ; search, GPU allowlist, validation nội bộ, fold, seed, mask, E0, metric và latency giữ
  nguyên; chạy lại 15 cell AutoTS; 15 cell v1 chuyển bằng `git mv` sang `superseded/autots_raw_price/`.

## 1. Việc đã chạy thật

| Bước | Lệnh / commit | Kết quả |
|---|---|---|
| Download | `python -m src_OB download --config configs/orderbook_zenodo.json` | tar 308.618.431 B, md5 `58507a0f…` khớp pin |
| Prepare | `python -m src_OB prepare …` (code c68fc45) | `data/orderbook/prepared_zenodo_20046390/` replay_version 2 |
| Data report | `python -m src_OB data-report …` | **READY**, 5 fold (`DATA_REPORT.md`) |
| Checker trước train | agent `checker` | không ERROR; Z-W1 đã sửa |
| Train lần 1 | `run_meta/run_train.sh` (code d8797c5), 02:46:25Z | 15 cell completed; dừng ở fold1/autots/h60s (mục 3) |
| Train lần 2 | `run_train.sh --resume` (code 688fefd), 02:58:19Z | skip 15 cell; SIGSEGV ở AutoTS candidate 9 (mục 3) |
| Train lần 3 | `run_train.sh --resume` (code 2a5a1c3), 03:02:28Z | 105 cell còn lại; `TRAIN_EXIT=0` 16:18:14Z; summarize 0 |
| Checker cuối run | agent `checker` (ZF) | không ERROR; 2 WARN về diễn giải report (đã sửa) |
| AutoTS v2: sửa + chuyển v1 | e8ca931 (adapter + README), 37aae35 (`git mv` 15 cell v1) | tree `src_OB/configs/src/p0` sạch |
| AutoTS v2 lần 1 | `run_train.sh --models autots` (code 37aae35), 17:53:30Z | 7 cell completed; lỗi pandas ở fold3/autots/h120s, `TRAIN_EXIT=1` 18:06:17Z (mục 3) |
| AutoTS v2 lần 2 | `run_train.sh --models autots --resume` (code 6108a1d) | skip 7; 8 cell còn lại; `TRAIN_EXIT=0` 18:29:34Z; `SUMMARIZE_EXIT=0` 18:29:35Z |
| Checker AutoTS v2 | agent `checker` (ZA) | không ERROR; 1 WARN (report còn mô tả v1, đã sửa ở đây), 6 INFO |

Lệnh train đúng như yêu cầu: `P0_OB_VAST=1 CUDA_VISIBLE_DEVICES=0 python -m src_OB train --config
configs/orderbook_zenodo.json` (launcher thêm `HF_HOME` ghi được, `--resume` khi recovery và `--models autots` cho lượt
AutoTS v2; mask origin chung vẫn lấy theo danh sách `models` của config). Log: `logs/train.log`, `logs/summarize.log`;
tmux `ob:ztrain`, `ztrain2`, `ztrain3`, `zautots`, `zautots2`.

Thời gian đo trong run thật: bốn family cây vài giây/cell; LSTM ~15 s/cell; AutoTS v2 ~2 phút/cell; TimesFM zero-shot
~3 phút/cell; TimesFM LoRA ~47 phút/cell (5 epoch, GPU ~61%, process 100% một lõi CPU vì overhead launch khi
forward/backward). Để latency p95/p99 không bị nhiễu, run dùng một process duy nhất trên GPU; epoch/batch/context giữ
nguyên.

Không chạy smoke/canary/test/probe/trial fit/benchmark/warmup riêng. Replay tham số AutoTS theo seed (mục 3) chỉ gọi
bộ sinh tham số ngẫu nhiên, không đọc data và không fit. Checker load checkpoint để đọc config, không fit/predict.

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
   lần 1 là `linear_tree=True`, tức **attempt 1 đã có một candidate fit trên CPU mà guard cũ không phát hiện** (suy từ
   replay seed và chuỗi cảnh báo trong lib; checker ZF-I5). Attempt đó không hoàn tất, không kết quả nào của nó được
   dùng; nó được giữ làm bằng chứng tại `attempts/fold1/autots/h60s/attempt1/`.
   Sửa (688fefd): giống XGBoost `gblinear → gbtree`, adapter AutoTS đặt `linear_tree=False` và đổi GOSS thành `gbdt`
   (luồng random theo seed không đổi). `GPURegressor` kiểm tham số trước fit, rồi đọc config hiệu lực trong model text
   sau mỗi fit LightGBM (`device_type`, `linear_tree`, `data_sample_strategy`, objective) và dừng job nếu không phải CUDA.
2. **Lần 2 — SIGSEGV với `max_bin` 1000.** `--resume` skip đúng 15 cell và chuyển attempt 1. AutoTS candidate 3 và 6
   (`max_bin` 255) fit CUDA xong; candidate 9 là LightGBM đầu tiên có `max_bin` 1000 (các tham số khác: `lambda_l1` 1,
   `min_data_in_leaf` 30, lr 0,01) thì process chết với SIGSEGV (exit 139), không có traceback. `max_bin > 255` đưa
   learner CUDA sang đường bin 16-bit và best-split dùng global memory, trong khi mọi fit CUDA thành công đều dùng
   đường 8-bit. `failed.json` được session ghi hậu kiểm vào cell trước khi resume chuyển thành `attempt2`.
   Sửa (2a5a1c3): allowlist giới hạn `max_bin ≤ 255`; launcher bật `PYTHONFAULTHANDLER=1`. Lần 3 không crash nữa.
3. **AutoTS v2 lần 1 — pandas từ chối weighted sample.** Ở fold3/autots/h120s, `NewGeneticTemplate` của AutoTS 1.0.4 gọi
   `DataFrame.sample(max_results, weights=log-rank, replace=False)` khi sinh generation 3; pandas 3.0.5 raise
   `ValueError: Weighted sampling cannot be achieved with replace=False` vì `size·max(p) > 1` (`core/sample.py:154`),
   điều pandas cũ (mà AutoTS được viết cho) không kiểm. Lỗi tất định theo seed. Nguyên nhân là AutoTS 1.0.4 không
   tương thích với pandas 3, không nằm trong phép center; v2 chạm tới trường hợp này vì điểm số khác làm pool template ở
   generation 3 khác v1.
   Sửa (6108a1d): trong scope search, `pandas.core.sample.sample` được thay tạm; đúng trường hợp sẽ raise thì dùng cách
   rút của pandas cũ (`random_state.choice(n, size, replace=False, p)`), mọi lời gọi khác đi đường pandas y hệt; khôi
   phục trong `finally`. Budget và trọng số search không đổi; attempt lỗi giữ ở `attempts/fold3/autots/h120s/attempt1/`.
4. **Recovery.** `train --resume` (thêm ở 688fefd) bỏ qua cell completed đúng config/revision (khác thì dừng) và chuyển
   attempt dở dang sang `attempts/<fold>/<model>/<h>/attemptN`; cell chạy lại ghi `train_code` và `prior_attempts`.
   Checker ZF-P2: không cell completed nào bị ghi đè.

## 3b. AutoTS: v1 (giá thô) → v2 (log-return quanh origin)

**v1 (code 2a5a1c3, nay ở `superseded/autots_raw_price/`).** Adapter hồi quy **mức giá thô** trên grid h giây bằng model
cây (không `normalize_window`, không transformer). Cây không bất biến theo mức giá: dự báo bị kẹp quanh các mức giá dày
đặc trong FIT, khi giá VAL ra khỏi khoảng đó thì thành ngoại suy. Checker ZF-W1 đọc cả 15 cell:

| fold | VAL so với khoảng giá FIT/dự báo | bias dự báo | RMSE so với E0 |
|---|---|---|---|
| fold1 | 100% VAL dưới min FIT; dự báo 27.273–27.350 | +493 … +513 | gain −20,5 … −32,0 |
| fold2 | VAL nằm trong vùng giá dày của FIT | ≈ −3 | vẫn 1,8–2,4× E0 |
| fold3 | 62–70% VAL cao hơn dự báo lớn nhất (≤ 28.186) | −224 … −245 | gain −1,9 … −4,7 |
| fold4 | bias nhỏ | −11 … −20 | vẫn 2,4–3,2× E0 |
| fold5 | 96–97% VAL cao hơn dự báo lớn nhất (≤ 28.818; FIT max 29.859, q99 28.662) | −922 … −966 | gain −23,5 … −36,8 |

**v2 (code e8ca931 → 6108a1d, user duyệt).** `CenteredLogReturn` bọc GPU estimator: cửa sổ native của AutoTS (W giá
oldest→newest, rồi regressor) được đổi thành log(P_window) − log(P_origin) với P_origin là giá mới nhất của cửa sổ; target
log(P(t+h)/P_origin); predict trả P_origin·exp(ŷ). Đây là phép biến đổi cố định, không phải learned transform; không
center toàn chuỗi; regressor không đổi. Bảng `df`, validation và chấm điểm nội bộ của AutoTS vẫn ở giá thô, nên
selection vẫn theo raw-price RMSE. `selected_model.json`/`model.joblib` ghi `target: log_return_centered_at_origin`,
`adapter_version: 2`.

| h | gain v1 theo fold (fold1 … fold5) | gain v2 theo fold (fold1 … fold5) |
|---|---|---|
| 60 | −31,96 / −1,40 / −4,66 / −2,14 / −36,83 | −0,0036 / −0,0322 / −0,0045 / +0,0005 / −0,0004 |
| 120 | −23,64 / −0,93 / −2,72 / −1,44 / −29,32 | −0,0100 / +0,0001 / +0,0008 / −0,0043 / −0,3336 |
| 180 | −20,49 / −0,78 / −1,93 / −2,18 / −23,54 | −0,0082 / −0,0013 / −0,0024 / +0,0035 / −0,1095 |

Search v2: mỗi cell 27 candidate × 3 vòng validation, 0 exception; model được chọn gồm cả LightGBM và XGBoost, window
2–90. Checker ZA: 27 candidate (12 ban đầu + 5 × 3
generation; fold5 h180 còn 26 vì AutoTS loại trùng) × 3 vòng, 0 exception, 0 vi phạm allowlist; model được chọn
đúng là argmin RMSE nội bộ ở 15/15; tái lập split khớp, target nội bộ cuối ≤ 23:59 ngày train_end (ZA-P4). Chọn 7
XGBoost + 8 LightGBM, trong đó 5 dart (fold3 h120, fold4 h60/h120/h180, fold5 h60), config hiệu lực đều cuda; điểm của
candidate dart ở v2 cùng khoảng gbdt, không còn lệch như ZF-W2 của v1 (ZA-I1). Nhiều model được chọn dự báo gần như
hằng số (std dự báo 2e-8…1,6e-4 so với std log-return thực 3,6e-4…3,8e-3; tương quan −0,05…+0,10): đây là hệ quả
của target log-return không scale theo thiết kế đã duyệt, không phải lỗi (ZA-I2). Các candidate chỉ khác nhau ở
`fillna` cho cùng một dự báo vì adapter đọc giá từ raw timeline, nên tốn budget nhưng không rò rỉ (ZA-I4).

## 4. Provenance

- Prepared: replay_version 2, code_commit c68fc45, `code_uncommitted_paths` rỗng, `config_sha256` `78aa19…` (sha của
  config JSON đã resolve). `run_meta/environment.json` ghi `2995bd…` = sha trên bytes của file config (checker Z-I6).
- Train code theo cell (`run.json` → `train_code`, tree sạch ở mọi cell): d8797c5 cho 15 cell fold1 lgbm/xgb/cat/xgbrf/lstm
  (lần 1); 2a5a1c3 cho 90 cell cat/lgbm/lstm/xgb/xgbrf/tfm còn lại; AutoTS v2: 37aae35 cho 7 cell (fold1 ×3, fold2 ×3,
  fold3 h60) và 6108a1d cho 8 cell. Diff 37aae35 → 6108a1d chỉ là shim pandas (mục 3.3), chỉ tác động lần rút sẽ raise;
  7 cell đầu không gặp trường hợp đó. Adapter code của 37aae35 giống e8ca931 (37aae35 chỉ di chuyển thư mục kết quả).
- `prior_attempts`: fold1/autots/h60s liệt kê `attempt1`, `attempt2` (hai lần lỗi thời v1, trước khi sửa GPU allowlist);
  fold3/autots/h120s liệt kê `attempt1` (lỗi pandas của v2). `run.json` của mọi cell giữ `status: "started"`; hoàn tất
  được đánh dấu bằng `completed.json`. 15 cell lần 1 không có key `prior_attempts` (ZF-I1).
- TimesFM zero-shot không có checkpoint riêng: repo `google/timesfm-2.5-200m-pytorch`, revision `1d952420…` được pin trong
  `src_OB/timesfm.py` và `logs/timesfm_checkpoint_download.log`. LoRA lưu adapter + OF head + scaler + repo/revision trong
  `adapter.pt` (ZF-I6, ZF-P8). AutoTS v2 lưu `CenteredLogReturn` (bọc GPU estimator) trong `model.joblib`.
- GPU theo config hiệu lực đọc từ checkpoint (ZF-P6 và ZA-P3): LightGBM `device_type cuda`, `linear_tree 0`,
  `max_bin 255`; XGBoost `cuda:0` hist; CatBoost `task_type GPU`; LSTM/TimesFM trên cuda theo code.
- Môi trường: RTX 3090 24 GB, driver 595.84, CUDA 12.8; torch 2.11.0+cu128, LightGBM 4.7.0 (CUDA, NCCL shared),
  XGBoost 3.4.1 (CUDA), CatBoost 1.2.10 (GPU), timesfm 2.0.2, AutoTS 1.0.4, pandas 3.0.5 (`run_meta/environment.json`,
  `pip_freeze.txt`).

## 5. Kết quả

**120/120 cell completed.** Tổng thời gian fit + inference đo trong cell: 13,2 h, trong đó LoRA 11,8 h, zero-shot
0,7 h, AutoTS v2 0,5 h, LSTM 0,1 h. Nguồn: `summary/by_model_horizon.csv`, `summary/per_fold_per_horizon.csv` (sinh lại
sau AutoTS v2).

Quy ước: RMSE/MAE trên raw mid price (USDT). `rmse_gain_vs_e0 = 1 − RMSE/RMSE_E0` (âm là tệ hơn E0);
`r2_os_vs_e0 = 1 − MSE/MSE_E0`. E0 = giá tại origin (zero return), cùng tập origin với mọi family. "mean" là trung bình
đều theo 5 fold; "pooled" gộp theo số prediction (fold2 chỉ có 7.621 origin, fold3/fold5 khoảng 31k), nên hai cách có
thể ngược dấu. Latency là inference thật, batch 1, lấy từ các request mẫu; "max" là giá trị lớn nhất quan sát được,
không phải giới hạn cứng.

| model | h | RMSE mean | gain mean (min … max) | pooled RMSE / E0 | pooled gain | R² OS mean | p95 / p99 / max ms |
|---|---|---|---|---|---|---|---|
| tfm_zero_shot | 60 | 25,32 | −0,025 (−0,052 … +0,010) | 33,59 / 33,52 | −0,002 | −0,051 | 82,2 / 83,1 / 241,2 |
| tfm_zero_shot | 120 | 35,96 | −0,030 (−0,064 … +0,016) | 49,89 / 50,00 | **+0,002** | −0,061 | 82,5 / 83,7 / 98,4 |
| tfm_zero_shot | 180 | 44,90 | −0,040 (−0,095 … +0,035) | 63,50 / 64,41 | **+0,014** | −0,084 | 82,6 / 83,5 / 98,4 |
| autots (v2) | 60 | 25,09 | −0,008 (−0,032 … +0,0005) | 33,64 / 33,52 | −0,004 | −0,016 | 2,7 / 2,8 / 5,5 |
| autots (v2) | 120 | 37,63 | −0,069 (−0,334 … +0,0008) | 52,19 / 50,00 | −0,044 | −0,161 | 2,7 / 2,9 / 5,6 |
| autots (v2) | 180 | 45,26 | −0,024 (−0,110 … +0,0035) | 65,34 / 64,41 | −0,014 | −0,050 | 2,8 / 2,9 / 5,6 |
| xgbrf | 60 | 25,21 | −0,004 (−0,044 … +0,031) | 33,91 / 33,52 | −0,012 | −0,009 | 2,9 / 3,0 / 17,9 |
| xgbrf | 120 | 36,66 | −0,035 (−0,130 … +0,008) | 51,36 / 50,00 | −0,027 | −0,073 | 2,9 / 3,0 / 18,0 |
| xgbrf | 180 | 46,53 | −0,057 (−0,146 … −0,004) | 66,55 / 64,41 | −0,033 | −0,121 | 2,8 / 2,9 / 18,0 |
| cat | 60 | 25,40 | −0,010 (−0,052 … +0,044) | 34,15 / 33,52 | −0,019 | −0,022 | 0,75 / 0,77 / 4,5 |
| cat | 120 | 36,47 | −0,034 (−0,092 … −0,002) | 50,94 / 50,00 | −0,019 | −0,070 | 0,76 / 0,77 / 3,8 |
| cat | 180 | 46,43 | −0,059 (−0,118 … −0,011) | 66,30 / 64,41 | −0,029 | −0,124 | 0,76 / 0,77 / 3,7 |
| lgbm | 60 | 25,68 | −0,024 (−0,103 … +0,032) | 34,41 / 33,52 | −0,027 | −0,050 | 0,81 / 0,92 / 6,6 |
| lgbm | 120 | 37,75 | −0,072 (−0,235 … −0,010) | 52,42 / 50,00 | −0,049 | −0,156 | 0,80 / 0,91 / 6,6 |
| lgbm | 180 | 48,84 | −0,121 (−0,414 … −0,009) | 68,71 / 64,41 | −0,067 | −0,281 | 0,78 / 0,84 / 6,6 |
| xgb | 60 | 26,61 | −0,076 (−0,186 … −0,022) | 35,26 / 33,52 | −0,052 | −0,161 | 1,9 / 2,0 / 222,7 |
| xgb | 120 | 40,39 | −0,185 (−0,542 … +0,001) | 54,37 / 50,00 | −0,088 | −0,440 | 1,9 / 2,0 / 12,6 |
| xgb | 180 | 50,15 | −0,169 (−0,379 … −0,023) | 69,65 / 64,41 | −0,081 | −0,384 | 1,9 / 2,0 / 161,6 |
| lstm | 60 | 28,85 | −0,188 (−0,254 … −0,067) | 37,43 / 33,52 | −0,117 | −0,417 | 0,34 / 0,35 / 0,95 |
| lstm | 120 | 46,79 | −0,417 (−0,821 … −0,072) | 60,40 / 50,00 | −0,208 | −1,075 | 0,34 / 0,35 / 1,2 |
| lstm | 180 | 56,91 | −0,392 (−0,839 … −0,023) | 75,44 / 64,41 | −0,171 | −1,008 | 0,34 / 0,35 / 1,2 |
| tfm_lora | 60 | 32,70 | −0,309 (−0,394 … −0,143) | 44,45 / 33,52 | −0,326 | −0,721 | 94,7 / 95,5 / 111,6 |
| tfm_lora | 120 | 46,93 | −0,345 (−0,479 … −0,241) | 65,81 / 50,00 | −0,316 | −0,814 | 94,5 / 95,6 / 103,3 |
| tfm_lora | 180 | 61,12 | −0,472 (−0,796 … −0,256) | 84,16 / 64,41 | −0,307 | −1,208 | 94,2 / 95,4 / 111,1 |

Đọc kết quả:
- Không family nào thắng E0 một cách nhất quán ở 60–180 s. **13/120 cell** có gain dương: autots v2 fold4 h60 (+0,0005),
  fold2 h120 (+0,0001), fold3 h120 (+0,0008), fold4 h180 (+0,0035); cat fold2 h60 (+0,044); lgbm fold2 h60 (+0,032);
  xgbrf fold1 h60 (+0,012), fold2 h60 (+0,031), fold2 h120 (+0,008); xgb fold3 h120 (+0,001); tfm_zero_shot fold3 cả
  ba horizon (+0,010, +0,016, +0,035).
- TimesFM zero-shot (chỉ dùng chuỗi mid) gần E0 nhất theo pooled (dương nhẹ ở h120/h180 nhờ fold3 biến động lớn, E0
  RMSE 54–110), nhưng trung bình theo fold vẫn âm ở cả ba horizon (ZF-I2). Chưa đủ để kết luận có lợi thế.
- AutoTS v2 đứng thứ hai theo pooled ở cả ba horizon và gần E0 như nhóm cây (gain v2 −0,0004 … −0,03 ở 13/15 cell),
  thay vì sai 1,8–38× E0 như v1. Hai cell kém rõ là fold5 h120 (−0,334) và fold5 h180 (−0,110), kéo mean h120/h180
  xuống. Vì nhiều model v2 dự báo return gần như hằng số (ZA-I2), "gần E0"
ở đây chủ yếu do dự báo gần zero-return. fold5 h120: 279/31.388 dự báo có |ŷ| > 0,5% (lớn nhất 2,2%), dồn vào 10-20
quanh 08:00, chỉ 40% đúng dấu, chiếm 73% phần SSE vượt E0 và không có bias trung bình; đây là variance của XGBoost
W=5, không phải lỗi code (ZA-I3).
- R² trên raw price cao ngay cả với E0 (fold1 h60: R² của E0 ≈ 0,976; `E0_R2` có trong `per_fold_per_horizon.csv`), nên
  R² 0,95–0,99 chủ yếu phản ánh giá tự tương quan. Chỉ số cần đọc là gain và R² OS so với E0; với bốn family cây dùng
  OF/OFI + timing, hai chỉ số này gần 0 và thường âm, trong đó xgbrf và cat gần E0 nhất ở h60.
- LSTM và TimesFM LoRA tệ hơn E0 ở mọi fold và horizon; LoRA tệ hơn zero-shot ở cả 15/15 cell. Train loss chuẩn hóa
  của LoRA còn 0,019–0,078 (≈ 1 tương ứng zero-return) trong khi sai số VAL tăng, tức overfit FIT. Nguyên nhân khả dĩ:
  target của các origin chồng lấp mạnh (origin cách nhau trung vị 2,5 s, horizon 60–180 s). Epoch cố định theo config,
  không early-stop hay tune bằng outer VAL (ZF-I4).
- Latency (ZF-I3): request ở vị trí 0 luôn nằm trong mẫu batch 1, nên max quan sát thường là lần gọi đầu (ví dụ xgb
  fold1 h60: max 222,7 ms, p99 1,89 ms). Batch cấu hình 32 nhưng mỗi lần gọi batch thực tế có 7–30 prediction, vì
  batch bị cắt tại origin mẫu kế tiếp; AutoTS gọi API từng origin. Latency batch ghi riêng trong `latency.json`.
- Giới hạn của data (mục 2): OF là flow giữa các snapshot REST ~1,24 s, không phải flow từng message; FIT hiệu dụng
  ~7,9 ngày; chỉ 21 ngày của tháng 10/2023, một chế độ thị trường duy nhất.

## 6. Quyết định còn cần từ user

Không còn. Quyết định về target AutoTS đã được user duyệt và thực hiện (mục 0, 3b).

## 7. Artifact và backup

- `experiments/orderbook_zenodo/`: `fold{1..5}/<model>/h{60,120,180}s/` (120 cell hiện hành, AutoTS = v2),
  `superseded/autots_raw_price/fold{1..5}/h*s/` (15 cell AutoTS v1 + README), `attempts/` (fold1/autots/h60s
  attempt1–2 thời v1; fold3/autots/h120s attempt1 thời v2), `summary/`, `logs/`, `run_meta/`, `DATA_REPORT.md`,
  `CHECKER_FINDINGS.md`, `BACKUP_STATUS.md`.
- Toàn bộ đã push lên `origin/OB` (LFS cho parquet/joblib/pt); chi tiết commit trong `BACKUP_STATUS.md`. Raw tar và
  checkpoint pretrained TimesFM không commit (tải lại được theo md5/revision đã pin).

## 8. Figures (hậu kỳ, 2026-09-11)

- User yêu cầu đọc `docs/visualize.txt`, nhưng file này không có trong repo (không có ở nhánh `origin/OB`, trong lịch sử
  git hay trên đĩa). Figure được làm theo yêu cầu trong session: khoảng 30 ảnh chồng TimesFM và AutoTS, mỗi ảnh đủ
  t → t+1 → t+2 → t+3, heatmap nếu đủ số liệu, chỉ dùng artifact đã lưu. Quy ước lấy từ `docs/RESEARCH_PLAN.md` §7.3:
  actual đen, E0 xám nét đứt, màu/marker cố định cho mỗi model, heatmap diverging dùng chung một thang.
- Lệnh: `python -m src_OB visualize --config configs/orderbook_zenodo.json` (`src_OB/visualize.py`). Lệnh không train,
  không inference, không dùng GPU, và dừng nếu origin/actual/E0 khác nhau giữa các family.
- `figures/paths/`: 30 ảnh, mỗi ảnh một origin. Mỗi ngày trong 10 ngày VAL lấy origin chung đầu tiên trong các giờ
  04:00/12:00/20:00 UTC. Mỗi ảnh chồng TimesFM zero-shot, TimesFM LoRA, AutoTS v2, actual, E0 và mid thô giữa các mốc;
  nhãn ghi lỗi tuyệt đối theo từng horizon.
- `figures/heatmap_fold_horizon.png` (5 fold × 3 h, lấy từ summary) và `figures/heatmap_day_horizon.png` (10 ngày
  VAL × 3 h; mỗi ngày có 2.999–20.427 origin, tính lại từ predictions). Kèm `figures/index.md`, `origins.csv`,
  `day_gains.csv`.
- Những gì thấy trên figure (chỉ để nhìn, kết luận vẫn theo metric §5):
  - AutoTS v2 gần như trùng đường E0, vì model dự báo return xấp xỉ 0 (ZA-I2). Nó chỉ kém rõ ở 10-20 và 10-21 tại
    h120/h180 (fold5, ZA-I3).
  - TimesFM zero-shot dao động quanh E0: dương ở 10-16 (cả ba horizon) và 10-12 (h60/h120), âm rõ ở 10-13, 10-15
    và 10-20.
  - TimesFM LoRA đỏ đậm ở mọi ô.
