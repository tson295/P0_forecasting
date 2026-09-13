# RUN_REPORT — phase tfm_autots trên Vast (BTC 1m hai năm)

Trạng thái bản ghi này: **ĐANG CHẠY** — viết trong lúc stage cuối `autots-search` còn chạy (bắt đầu 2026-09-13 17:38:20 UTC).
Các mục đánh dấu `CHỜ` sẽ được điền từ artifact thật khi phase kết thúc. Không có số nào ở đây là ước lượng hay suy đoán;
mọi con số lấy từ log/artifact của lượt chạy thật `experiments/tfm_autots_sessions/run_20260911T190135Z_Jmb166`.

## 1. Data coverage thực tế

| Nguồn | Bar | Khoảng | Bytes | SHA256 |
|---|---|---|---|---|
| HF 1m (`data/BTC_1m_2y.csv`) | 1.051.201 | 2024-09-03 16:29 → 2026-09-03 16:29 UTC | 101.766.374 | `559ce040…e31f8097` |
| LF 5m (`data/BTC_5m_2y.csv`) | 210.239 | 2024-09-03 16:35 → 2026-09-03 16:25 UTC | 21.146.273 | `0e5fb9ad…40f52fef` |

Cả hai khớp `data/data_checksums_2y.json` và `data/BTC_5m_2y.derivation.json` (kiểm bằng `sha256sum` sau `git lfs pull`).
1m là timeline target; 5m chỉ sinh feature phụ. Không tải và không dùng Order Book.
Đường chạy thật gọi `load_store(verify=True)` (`src/p0/cli.py:184-186`), nên sai checksum sẽ dừng run chứ không ghi đè.

## 2. Môi trường và provenance

- Máy: 1× NVIDIA GeForce RTX 3090 24 GiB (sm_86), driver 580.159.03, 80 luồng CPU, 188 GiB RAM, không sudo.
- Code: commit `d9201e1`, `code_changes.patch` rỗng (không sửa code trong lúc chạy).
- `.venv` Python 3.11.16: torch 2.11.0+cu128, xgboost 3.2.0, cupy-cuda12x 14.2.0, autots 1.0.4, timesfm 2.0.2,
  numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.1, statsmodels 0.15.0. Không cài JAX/XReg.
- LightGBM 4.7.0 build từ sdist với `USE_CUDA=ON`, `BUILD_WITH_SHARED_NCCL=ON`, `CMAKE_CUDA_ARCHITECTURES=86`
  (CUDA toolkit 12.8.93, gcc 13.3, cmake 3.28.3). `lib_lightgbm.so`: cubin sm_86, NEEDED `libnccl.so.2`, cudart tĩnh.
  Chi tiết và log gốc: `../tfm_autots_sessions/SETUP_ENV.md`, `../tfm_autots_sessions/setup_logs/`.
- Checkpoint TimesFM tải khi model load thật: `google/timesfm-2.5-200m-pytorch`, revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`
  (đúng pin ở `src/p0/models_tfm.py:32`), lưu ngoài `experiments/` trong `HF_HOME`.
- Không chạy test/smoke/canary/pytest/probe fit/benchmark/warmup/latency replay ở bất kỳ bước nào của goal này.

## 3. Stages và runtime thật

| Stage | Trạng thái | Thời gian (`phase_progress.json`) |
|---|---|---|
| lock-s0 | xong | 37,0 s |
| loop:tfm | xong | 118.394,6 s (32,9 h) |
| tfm-final | xong | 0,0 s — tái dùng predictions confirmation cùng seed, không fit lại |
| loop:autots_wr | xong | 44.336,1 s (12,3 h) |
| loop:autots_mr | xong | 5.026,5 s (1,4 h) |
| autots-search | đang chạy | CHỜ |
| summary | chưa | CHỜ |

S0 sau collision handling: TimesFM S0 = ∅ (baseline feature-free); AutoTS WR = 72 B0* + 21 ext; AutoTS MR = 72 B0* + 8 ext.
Candidate pool giữ nguyên 163 cột cho cả ba nhánh (không cột nào bị trừ do trùng; tương quan cao chỉ báo cáo).
Folds: rolling_spread 5 fold, FIT 120d / ES 5d / VAL 3d / purge 60m, TEST 30d **không** được đánh giá trong phase này.
Seeds: calib 8586, eval 8587/8588/8589, selection 8587 — đúng config đã khoá.

## 4. Kết quả

### 4.1 TimesFM (`wins/tfm.json`, `tfm_final.csv`)

- LoRA: 5 fit ES (calib seed) mỗi fit 6 epoch, `best_epoch = 1`, 5.171–5.310 s/fit; `fixed_epoch_TFM = 1`
  → 15 fit fixed-epoch (841–860 s/fit); confirmation §2.1b fit thêm 15 adapter ES cho ba eval seed.
  Tổng 35 adapter (`lora/*.pt`), 2.048.000/233.337.280 tham số trainable, context 512, batch 64.
- Forecast batch native: ~11,2–11,8 s cho ~4.300 origins mỗi fold (256 origins/batch), cache theo adapter/data/origins.
- ε = 0,3591 pp. Baseline TimesFM-LoRA native: MedianGain vs E0 = **−0,3307 pp** (thua E0).
- Add-one 163 candidate: tất cả KEEP (gain âm nhỏ vẫn nằm trong biên ε). `F*_raw` 163 KEEP / 0 DROP,
  MedianGain vs baseline S0 = **−3,4233 pp**. Prune PI giữ 160/163.
- Đại diện cuối: `tfm_lora_baseline` (hệ A, feature-free), MedianGain vs E0 = **−0,2424 pp**.
  Hệ B (LoRA + XReg 160 cột) thua A: MedianGain_B_vs_A = **−3,4279 pp**, WinRate 0,0, P10 −119,30, Worst −179,23.
- Gain vs E0 theo fold × horizon (pp): `[[−0,161, −0,2424, −0,4605], [−0,0177, −0,0966, −0,2757],
  [−0,6218, −1,1426, −1,5130], [−0,2288, −0,4059, −0,6077], [−0,0867, −0,1302, −0,2058]]` — âm ở mọi ô.

### 4.2 AutoTS WR (`wins/autots_wr.json`)

- Cache FIT-only: 256 cột covariate mỗi fold, 1,1–1,3 s/fold, `READY.json` ghi preprocessing 36,8 s; scaler FIT-only.
- Fit thật trên GPU: `autots_fits/autots_wr/*.json` ghi `prepared_fit.gpu_fit_seconds ≈ 29,5 s`
  tách khỏi `selection_seconds ≈ 0,14 s`; train_rows 172.735, train_columns 153;
  predict theo batch 256 origins × 3 bước, ~0,016–0,018 s/batch (`independent_origins_batched_v1`).
- `F*_raw` 162 KEEP / 1 DROP, MedianGain vs baseline S0 = **+0,1193 pp**. Prune PI giữ 90/162 cột mới.
- `F_win = prune` (confirmation F_raw vs F_pruned: MedianGain +0,0683 pp, ε = 0,1690) → F_best = 90 cột mới + 21 ext khoá.

### 4.3 AutoTS MR (`wins/autots_mr.json`)

- Cache READY 16:15:37. Fit MR rẻ hơn WR rất nhiều nhưng vẫn trên GPU: `gpu_fit_seconds ≈ 1,34 s`,
  train_rows 172.796, train_columns 80; prediction 3,13 s cho 4.317 origins (recursive 3 bước).
- `F*_raw` 33 KEEP / 130 DROP, MedianGain vs baseline S0 = **+0,3597 pp** (cao nhất trong ba nhánh).
  Prune PI giữ 21/33. `F_win = prune` (MedianGain +0,0137 pp, ε = 0,0050) → F_best = 21 cột mới + 8 ext khoá.

### 4.4 Bake-off `autots-search` và summary

Cấu hình: 4 template / 2 nhóm shift `['wr:60', 'mr']`, `num_validations = 10`, 2 frozen set × 5 fold.
Đơn vị đầu (`F_WR_best|wr:60`, 5 fold) xong 18:37:20, template thắng **WindowRegression** ở cả 5 fold;
template JSON lưu ở `autots_templates/` (5 file `best_…` + 5 file `all_…`).

Đường native của bake-off ghi fit record riêng: 5 record có `frozen_template` (đúng 5 fold của đơn vị đã xong),
`fit_timing_scope = "native preprocessing plus GPU fit"`, fit 5,6–56,1 s, prediction 0,32 s. Các record này có
`prepared_fit = null` vì không đi qua cache fit của feature-search, nên **không có `gpu_fit_seconds` tách riêng**
cho bake-off — chỉ có thời gian gồm cả native preprocessing. Không có dấu hiệu CPU fallback trong bất kỳ record nào.

Các đơn vị còn lại và hai bảng
`tfm_autots_per_fold_horizon.csv`, `tfm_autots_summary.csv`: **CHỜ**.

## 5. Lỗi đã gặp và cách xử lý

1. `HF_HOME=/workspace/.hf_home` của image thuộc `root:root 775`, user `ubuntu` không ghi được → truyền
   `HF_HOME=/home/ubuntu/.cache/huggingface` khi tạo tmux session. Chỉ đổi vị trí cache, không đổi code/config/data.
2. Push lần đầu hỏng vì `LFS Put … i/o timeout` khi upload lên github-cloud S3 (lỗi mạng, không phải quota/auth;
   SSH đã xác thực và remote vẫn là ancestor). Xử lý: `lfs.concurrenttransfers=3`, `activitytimeout=300`,
   `dialtimeout=60`, `tlstimeout=60` rồi retry → thành công 294 MB/76 object.
3. Không có lỗi runtime nào trong code của phase; không sửa file nào trong `src/p0` (code hash giữ nguyên suốt run).

## 6. Hạn chế của báo cáo này

- **Không có TEST holdout**: fold giữ TEST 30 ngày nhưng phase này không đánh giá — đúng phạm vi goal.
- **Không có champion/ensemble/model family khác**: `defer_champion=true`, chỉ TimesFM và hai nhánh AutoTS.
- **Không có metric latency**: phase tắt warmup/latency replay; các số thời gian là fit/predict theo batch thật.
  Thời gian batch **không phải** p95 single-request, timing cache-hit **không phải** native inference latency,
  và max quan sát được **không phải** hard bound.
- **R²/MAE**: chỉ đọc từ artifact thật khi có; không bịa metric chưa xuất, không gọi correlation r là R².
- Log text không in epoch cuối của mỗi ES fit vì `lora.py:188-190` `break` trước lệnh `log`; epoch đó vẫn nằm trong
  `curve` của JSON adapter. Đây là khác biệt hiển thị, không ảnh hưởng kết quả; không sửa vì sẽ đổi code hash.
- Cảnh báo pandas `divide by zero encountered in log` khi sinh feature là hành vi sẵn có (inf → NaN rồi lọc `isfinite`).
- NCCL: `lib_lightgbm.so` không có RPATH; trong process map có cả bản torch (2.28.9) lẫn bản hệ thống (2.31.2).
  LightGBM một GPU không khởi tạo NCCL collectives.

## 7. Trạng thái git/push

| Commit | Nội dung | Push |
|---|---|---|
| `02fe142` | setup provenance + lock-s0 + CHECKER_FINDINGS mốc 1 | đã push |
| `32b1bb3` | artifact stage TimesFM (wins, lora/, CSV, runs) | đã push (76 LFS object, 294 MB) |
| `1105a38` | kết quả AutoTS WR (wins, keepdrop/prune, calib, runs) | đã push |
| `80554b6` | estimator states + cache fold của WR (3,3 GB) | đang push |
| MR + summary + report | CHỜ | CHỜ |

`experiments/15d` không bao giờ được stage: 71 file ở đó hiện "modified" chỉ vì `.gitattributes` áp LFS lên các blob
đã commit thường (blob hash = index hash, 0 dòng thay đổi). Mọi commit dùng path tường minh, không `git add -A/-u`.
Băng thông upload của instance đo được 64–455 KB/s; nếu các khối nhiều GB không kịp đẩy thì sẽ ghi PUSH_PENDING
đúng từng path kèm số đo, giữ commit và LFS object ở local, không tuyên bố đã backup ngoài máy.
