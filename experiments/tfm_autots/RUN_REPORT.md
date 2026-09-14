# RUN_REPORT — phase tfm_autots trên Vast (BTC 1m hai năm)

Lượt chạy thật: `experiments/tfm_autots_sessions/run_20260911T190135Z_Jmb166`, bắt đầu 2026-09-11 19:01:35 UTC,
kết thúc 2026-09-14 00:26:46 UTC (≈ 53 h 25 min), `exit_code.txt = 0`, `phase_progress.json: status = completed`.
Mọi số trong báo cáo lấy từ log/artifact của chính lượt chạy này. Không chạy test/smoke/canary/pytest/probe fit/
synthetic/benchmark/warmup/latency replay ở bất kỳ bước nào. Không đánh giá TEST holdout, không champion/ensemble,
không family nào khác ngoài TimesFM và hai nhánh AutoTS.

## 1. Data coverage thực tế

| Nguồn | Bar | Khoảng | Bytes | SHA256 |
|---|---|---|---|---|
| HF 1m (`data/BTC_1m_2y.csv`) | 1.051.201 | 2024-09-03 16:29 → 2026-09-03 16:29 UTC | 101.766.374 | `559ce040…e31f8097` |
| LF 5m (`data/BTC_5m_2y.csv`) | 210.239 | 2024-09-03 16:35 → 2026-09-03 16:25 UTC | 21.146.273 | `0e5fb9ad…40f52fef` |

Khớp `data/data_checksums_2y.json` và `data/BTC_5m_2y.derivation.json` (kiểm `sha256sum` sau `git lfs pull`).
1m là timeline target, 5m chỉ sinh feature phụ; không tải và không dùng Order Book.
Đường chạy thật gọi `load_store(verify=True)` (`src/p0/cli.py:184-186`) nên sai checksum sẽ dừng run.

Split thực tế trong run (đọc từ artifact): FIT 172.797 bar; inner ES 7.197 bar (5 ngày); residual calibration suffix
7.197 bar (5 ngày); purge 3.600 s giữa các phần; VAL 4.300–4.317 origin mỗi fold (3 ngày); 5 fold rolling_spread.
TEST 30 ngày tồn tại trong split nhưng **không** được đánh giá trong phase này.
Seeds: calib 8586, eval 8587/8588/8589, selection 8587 — đúng config đã khoá.

## 2. Môi trường và provenance

- Máy: 1× NVIDIA GeForce RTX 3090 24 GiB (sm_86), driver 580.159.03, 80 luồng CPU, 188 GiB RAM, không sudo.
- Code: commit `d9201e1`, `code_changes.patch` rỗng — không sửa file nào trong `src/p0` trong suốt run.
  Code hash `03d7a4a8e4b34ff25f89cc51f397e898604e5eee0cf42fe96c99532e42c8cec2`, config hash `822935ae5e8f`.
- `.venv` Python 3.11.16: torch 2.11.0+cu128, xgboost 3.2.0, cupy-cuda12x 14.2.0, autots 1.0.4, timesfm 2.0.2,
  numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.1, statsmodels 0.15.0. Không cài JAX/XReg.
- LightGBM 4.7.0 build từ sdist: `USE_CUDA=ON`, `BUILD_WITH_SHARED_NCCL=ON`, `CMAKE_CUDA_ARCHITECTURES=86`
  (CUDA toolkit 12.8.93, gcc 13.3, cmake 3.28.3). `lib_lightgbm.so`: cubin sm_86, NEEDED `libnccl.so.2`, cudart tĩnh.
  Log build và pip: `../tfm_autots_sessions/setup_logs/`; tóm tắt: `../tfm_autots_sessions/SETUP_ENV.md`.
- Checkpoint TimesFM tải khi model load thật: `google/timesfm-2.5-200m-pytorch`, revision
  `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` (đúng pin `src/p0/models_tfm.py:32`), lưu ngoài `experiments/`.

## 3. Stages và runtime thật (`phase_progress.json`, khớp mốc thời gian trong `training.log`)

| Stage | Thời gian |
|---|---|
| lock-s0 | 37,0 s |
| loop:tfm | 118.394,6 s (32,9 h) |
| tfm-final | 0,006 s — tái dùng predictions confirmation cùng seed, không fit lại |
| loop:autots_wr | 44.336,1 s (12,3 h) |
| loop:autots_mr | 5.026,5 s (1,4 h) |
| autots-search | 24.489,9 s (6,8 h) — gồm cả bước `summarize_phase` ghi hai bảng tổng hợp |

Không có stage tên `summary` riêng: `summarize_phase()` chạy bên trong `autots-search`.

S0 sau collision handling: TimesFM S0 = ∅ (baseline feature-free); AutoTS WR = 72 B0* + 21 ext; AutoTS MR = 72 B0* + 8 ext.
Candidate pool 163 cột cho cả ba nhánh (không cột nào bị trừ; tương quan cao chỉ báo cáo).
`runs/` có 510 thư mục = 163 × 3 nhánh + 21 lần confirmation.

## 4. Kết quả

### 4.1 TimesFM (`wins/tfm.json`, `tfm_final.csv`)

- LoRA: 35 adapter `lora/*.pt` = 5 ES (calib seed 8586) + 15 fixed-epoch + 15 ES của confirmation (3 eval seed × 5 fold).
  Tổng 20 fit chế độ ES, mỗi fit 5.079,4–5.313,7 s; 15 fit fixed-epoch 836,6–865,0 s.
  Mọi ES fit đều chọn `best_epoch = 1` → `fixed_epoch_TFM = 1`. 2.048.000/233.337.280 tham số trainable, context 512, batch 64.
- Artifact kèm theo: 845 residual head, 885 runtime record, 55 forecast-cache entry.
  Forecast native theo batch: ~11,2–11,8 s cho ~4.300 origin mỗi fold (256 origin/batch).
- ε = 0,3591 pp. Baseline TimesFM-LoRA native: MedianGain vs E0 = **−0,3307 pp** (thua E0).
- Add-one 163 candidate: tất cả KEEP (gain âm nhỏ vẫn trong biên ε). `F*_raw` 163 KEEP / 0 DROP,
  MedianGain vs baseline S0 = **−3,4233 pp**. Prune PI giữ 160/163.
- Đại diện cuối: `tfm_lora_baseline` (hệ A, feature-free), MedianGain vs E0 = **−0,2424 pp**.
  Hệ B (LoRA + XReg 160 cột) thua A: MedianGain_B_vs_A = **−3,4279 pp**, WinRate 0,0, P10 −119,30, Worst −179,23.
  Mức Worst đó do **một fold bung**, không phải suy giảm đều: `tfm_final.csv` fold4 h1/h2 RMSE 128,6/183,0
  trong khi E0 fold4 là 45,9/65,9; các fold khác của hệ B bám sát baseline.

### 4.2 AutoTS WR (`wins/autots_wr.json`)

- Cache FIT-only: 256 cột covariate mỗi fold, 1,1–1,3 s/fold, `READY.json` (preprocessing 36,8 s), scaler FIT-only.
- Fit thật trên GPU (`autots_fits/autots_wr/*.json`, đường feature-search): `prepared_fit.gpu_fit_seconds`
  **29,09–71,05 s** (trung vị 48,4 s), tách khỏi `selection_seconds`; `train_columns` 153–315 tuỳ bước add-one;
  predict theo batch 256 origin × 3 bước, **0,0123–0,0566 s** mỗi batch (`independent_origins_batched_v1`).
- `F*_raw` 162 KEEP / 1 DROP, MedianGain vs baseline S0 = **+0,1193 pp**. Prune PI giữ 90/162 cột mới.
- `F_win = prune` (confirmation F_raw vs F_pruned: MedianGain +0,0683 pp, ε = 0,1690) → F_best = 90 cột mới + 21 ext khoá.

### 4.3 AutoTS MR (`wins/autots_mr.json`)

- Cache FIT-only 243 cột, READY 16:15:37. Fit MR rẻ hơn WR nhưng vẫn trên GPU:
  `gpu_fit_seconds` **0,585–1,403 s**, `train_columns` 80–116; prediction theo batch **0,117–2,95 s** mỗi batch
  (recursive 3 bước, mỗi origin có history riêng).
- `F*_raw` 33 KEEP / 130 DROP, MedianGain vs baseline S0 = **+0,3597 pp** (cao nhất trong ba nhánh).
  Prune PI giữ 21/33. `F_win = prune` (MedianGain +0,0137 pp, ε = 0,0050) → F_best = 21 cột mới + 8 ext khoá.

### 4.4 Bake-off `autots-search` và đại diện cuối (`autots_search.csv`)

4 template / 2 nhóm shift `['wr:60','mr']`, `num_validations = 10`, 2 frozen set × 5 fold → 4 đơn vị,
40 file template trong `autots_templates/` (4 đơn vị × 5 fold × `best_`/`all_`).

| Đơn vị | n_ext | Template thắng | MedianGain vs E0 @ selection_seed 8587 |
|---|---|---|---|
| F_WR_best \| wr:60 | 111 | WindowRegression | −0,3343 pp |
| **F_WR_best \| mr** | **111** | **MultivariateRegression** | **−0,2185 pp → ĐẠI DIỆN CUỐI** |
| F_MR_best \| wr:60 | 29 | WindowRegression | −0,5551 pp |
| F_MR_best \| mr | 29 | MultivariateRegression | −0,3447 pp |

Confirmation của đại diện cuối trên ba eval seed: MedianGain vs E0 = **−0,2185 pp**, ε = 0,0050 pp (đúng `eps_floor_pp`);
predictions của seed 8587 được tái dùng vì trùng seed selection, không fit lại ô trùng.

Đường native của bake-off ghi 30 fit record riêng: `fit_timing_scope = "native preprocessing plus GPU fit"`,
fit **4,24–74,76 s**, predict **0,129–4,009 s**, và `prepared_fit = null` — nên **không tồn tại `gpu_fit_seconds`
tách riêng cho bake-off**; báo cáo không trích số GPU thuần nào ở đây. Không record nào có dấu hiệu CPU fallback.

### 4.5 Hai bảng tổng hợp

`tfm_autots_summary.csv` (đơn vị của `rmse_gain_vs_e0` và `r2_os_vs_e0` là **phân số**, không phải pp;
aggregation = **trung bình không trọng số các metric theo fold**, không phải pooled; RMSE ở **giá thô**):

| model | h | RMSE (mean over seeds) | RMSE E0 | gain vs E0 | R² OS vs E0 |
|---|---|---|---|---|---|
| autots | 1 | 54,01955 | 53,93927 | −0,0010110 | −0,0020276 |
| autots | 2 | 77,50510 | 76,14044 | −0,0132191 | −0,0268387 |
| autots | 3 | 96,44707 | 92,89368 | −0,0280082 | −0,0578828 |
| tfm | 1 | 54,08653 | 53,93927 | −0,0022319 | −0,0044756 |
| tfm | 2 | 76,51473 | 76,14044 | −0,0040354 | −0,0081144 |
| tfm | 3 | 93,57405 | 92,89368 | −0,0061252 | −0,0123349 |

`tfm_autots_per_fold_horizon.csv` giữ 30 ô fold × horizon cho hai đại diện, kèm cột `aggregation` và `e0_status`
(tất cả `ok`, không ô nào có E0 = 0). R² OS tính bằng `1 − mean(MSE theo từng seed)/E0²` (`phase_tfm_autots.py:120`),
**không** bình phương mean-seed RMSE.

**Kết luận trung thực: không family nào thắng E0 ở giá thô trên VAL.** Cả 6 ô tổng hợp đều âm, R² OS âm ở mọi horizon,
và mức thua tăng theo horizon (AutoTS h3 −2,8 %). TimesFM thua ít hơn AutoTS ở h2/h3, AutoTS thua ít hơn ở h1.

**Độ phân tán theo seed của AutoTS bằng 0 do cấu trúc**: template đông lạnh ghim `random_state: 8587` và nhánh MR
không bootstrap, nên 10 lần refit confirmation ở seed 8588/8589 tái tạo đúng cùng một fit — `seed_rmse` giống hệt nhau
và `wins/autots_seed{0,1,2}.npz` trùng SHA256. Vì vậy hàng `autots` **không phải** trung bình ba seed độc lập.
Hai đại diện TimesFM và nhánh WR thì khác biệt thật theo seed.

## 5. Lỗi đã gặp và cách xử lý

1. `HF_HOME=/workspace/.hf_home` của image thuộc `root:root 775`, user `ubuntu` không ghi được → truyền
   `HF_HOME=/home/ubuntu/.cache/huggingface` khi tạo tmux session. Chỉ đổi vị trí cache, không đổi code/config/data.
2. Push đầu tiên hỏng vì `LFS Put … i/o timeout` lên github-cloud S3 (lỗi mạng, không phải quota/auth). Xử lý:
   `lfs.concurrenttransfers=3`, `activitytimeout=300`, `dialtimeout=60`, `tlstimeout=60` rồi retry → thành công.
   Lần push 3,4 GB sau đó upload xong 919 object nhưng bị remote đóng kết nối lúc cập nhật ref (exit 141);
   attempt kế tiếp chỉ cần cập nhật ref là xong.
3. Không có lỗi runtime nào trong code phase: `training.log` không có traceback/OOM, không có record `device: cpu`.

## 6. Metric chưa xuất và hạn chế

- **Không có MAE cho hai đại diện cuối**: `mae_cells` chỉ được điền cho các dòng search/confirmation trong `log.csv`,
  còn `loop_final`, `tfm_final`, `autots_search` để trống; `champion_log.csv` cũng trống cột MAE/RMSE.
- **Không có metric latency nào**: phase tắt warmup/latency replay, `latency_p95/p99/max` trống hoàn toàn.
  Các số thời gian trong báo cáo là fit/predict theo batch thật; **thời gian batch không phải p95 single-request**,
  **timing cache-hit không phải native inference latency**, và **max quan sát được không phải hard bound**.
- Không gọi bất kỳ correlation nào là R²; R² OS ở đây đúng công thức nêu trên.
- **Mức sử dụng GPU không kiểm được từ artifact**: artifact chỉ chứng minh backend CUDA (`device: cuda`,
  `gpu_fit_seconds` tách riêng ở đường feature-search) chứ không ghi utilisation. Trong phiên, `nvidia-smi` đọc
  73–100 % và 5,2–8,7 GiB VRAM lúc LoRA/AutoTS fit — đó là quan sát của session, không phải bằng chứng artifact.
- Log text không in epoch cuối của mỗi ES fit vì `lora.py:188-190` `break` trước lệnh `log`; epoch đó vẫn nằm trong
  `curve` của JSON adapter. Khác biệt hiển thị, không ảnh hưởng kết quả; không sửa vì sẽ đổi code hash.
- Cảnh báo pandas `divide by zero encountered in log` khi sinh feature là hành vi sẵn có (inf → NaN rồi lọc `isfinite`).
- NCCL: `lib_lightgbm.so` không có RPATH; process map có cả bản torch (2.28.9) lẫn bản hệ thống (2.31.2).
  LightGBM một GPU không khởi tạo NCCL collectives.
- TEST holdout, champion/ensemble và các family khác nằm ngoài phạm vi phase này.

## 7. Trạng thái git/push

| Commit | Nội dung | Push |
|---|---|---|
| `02fe142` | setup provenance + lock-s0 + CHECKER_FINDINGS mốc 1 | đã push |
| `32b1bb3` | artifact stage TimesFM (wins, lora/, CSV, runs) — 76 LFS object, 294 MB | đã push |
| `1105a38` | kết quả AutoTS WR (wins, keepdrop/prune, calib, runs) | đã push |
| `80554b6` | estimator states + cache fold của WR — 919 LFS object, 3,4 GB | đã push |
| `8e00a56` | kết quả AutoTS MR + bản nháp RUN_REPORT | đã push |
| `cbb9cb6` | đại diện AutoTS cuối, hai bảng summary, PHASE_REPORT, templates, log.csv, training.log, exit_code | đã push |
| `dde5539` | `autots_fits/*`, `autots_preprocess/autots_mr`, `series_preprocess` — 937 LFS object, ~4,0 GB | đã push |
| báo cáo cuối | RUN_REPORT (bản này) + CHECKER_FINDINGS mốc 2 + MEMORY | push ngay sau khi commit |

PUSH_STATUS (2026-09-14 02:24 UTC): toàn bộ output của phase đã nằm trên `origin/tfm_autots`; **không có path nào
PUSH_PENDING**. `dde5539` cần hai attempt: attempt 1 upload đủ 937 object rồi bị remote đóng kết nối lúc cập nhật ref
(exit 141), attempt 2 chỉ cập nhật ref. Commit báo cáo cuối được push ngay sau khi tạo; nếu push đó hỏng thì trạng thái
sẽ được ghi bằng một commit bổ sung có bằng chứng, không sửa lịch sử và không force push.

`experiments/15d` không bao giờ được stage: 71 file ở đó hiện "modified" chỉ vì `.gitattributes` áp LFS lên các blob
đã commit thường (blob hash = index hash, 0 dòng thay đổi); checker xác nhận **không có file `experiments/15d` nào**
trong dải `02fe142..HEAD`. Mọi commit dùng path tường minh, không `git add -A/-u`, không force push, không commit
secret/venv/checkpoint pretrained. Băng thông upload của instance đo được 64–455 KB/s, nên các khối nhiều GB
được tách commit và push nền theo từng stage.
