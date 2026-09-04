# MEMORY — trạng thái (update/replace, không append mâu thuẫn)

PHASE: **MIGRATION EXPANDED-DATA + PASS HIỆU CHỈNH XONG (code/config/test/doc) — chờ checks trên data thật + user unlock**, 2026-09-04
TRAINING: LOCKED

## Current Task

Vòng **expanded-data** (quyết định user 2026-09-03; pass hiệu chỉnh 2026-09-04). Trạng thái: **migration corrections complete**.
- Vòng 15 ngày (2026-08-31 → 09-01) = lịch sử, artifact + kết quả ở `experiments/15d/` (không sửa, không đổi tên).
- TimesFM hiện tại = **TimesFM-LoRA native làm baseline + XReg search trên C_short, KHÔNG B0\*** (không có nhánh tfm_b0).
- **Candidate_m = C_short \ overlap(C_short, S0_m)** tính riêng từng model (`experiments/full/s0/candidates_<m>.json`); tương quan cao chỉ báo cáo.
- Checker = **không tương tác**, finding ghi `experiments/<run>/checker_log.jsonl`; bất biến cứng do code ép (`hard_fail`).
- **Chưa chạy training vòng mới, chưa chạm TEST.** 139 unit test PASS, smoke synthetic PASS (lock-s0 → loop → final → sentinel → visualize).

## Exact Next Step

1. Đặt data đầy đủ vào `data/BTC_hf_1min_full.csv` + `data/BTC_lf_5min_full.csv` (LF phải phủ toàn bộ HF).
   `python run.py check-data --config configs/p0_full.json --write-checksums` → fold rolling §1.5 (cần ≥ 90 ngày).
2. `python run.py lock-s0 --config configs/p0_full.json` → `experiments/full/s0/` trên data thật (loop từ chối nếu audit_dataset_label khác config).
   (Đã chạy thử trên data 15 ngày qua `--data-config`: C_short 163, Candidate_m = 163 cho mọi model — 0 overlap với S0; near chỉ báo.)
3. Trên Vast: `bash scripts/vast_bootstrap.sh` → `scripts/vast_canary.py` + `scripts/canary_xreg_gpu.py` → `pytest -q -x` →
   `python scripts/checker_record.py --exp experiments/full --blocking` (không còn ERROR) → agent `checker` (không tương tác, ghi checker_log).
4. **User unlock rõ ràng** → `scripts/canary_lora.py` (1 fold × 1 epoch, adapter tạm, đo thời gian/VRAM) → `docs/VAST_SESSION_PROMPT.md`
   (thứ tự §8: loop lgbm → xgb → cat → tfm → tfm-final → xgbrf → autots_wr → autots_mr → autots-search → lstm → ensemble → final → visualize).

## Decisions (mới nhất trước)

- 2026-09-04 (user, pass hiệu chỉnh — ghi đè các điểm dưới của 2026-09-03): (a) **toàn bộ S0_m là khoá**: artifact `s0/<m>.json` ghi
  `locked_b0 == b0` (B0* 72 cột) và `locked_ext == ext` (F_old_m); `ColSet` có `locked_b0`/`locked_ext`, `without_b0` luôn từ chối; đọc được
  artifact cũ ({"b0","ext"} / {"locked"}). Tên: S0_m (xuất phát khoá) ≠ F_raw/F_pruned/F_best (kết quả tìm kiếm mới). (b) **Candidate_m chỉ trừ
  overlap với S0_m của CHÍNH model** (trùng tên / giá trị giống hệt cùng timestamp); **KHÔNG lọc toàn cục** theo B0-306 hay candidate cũ; khác lag ≠
  trùng; **tương quan cao = báo cáo, không bao giờ tự xoá** (`near_vs_s0`, `intra_short_near` chỉ chẩn đoán). Bản 2026-09-03 từng ghi sai rằng user
  bỏ Keltner ngắn vì corr ≥ 0.999999 — SAI, đã gỡ; Keltner ngắn được khôi phục. (c) **C_short dày ≤ 15'**: lưới {1,2,3,4,5,8,10,15} cho mọi họ, chỉ bỏ
  cửa sổ suy biến/không xác định về toán hoặc chính là candidate cũ §2.3; thêm họ Keltner `kcw_k`, PSAR cửa sổ reset `psar_{dir,logdist,age_log}_W`
  (W ∈ {2..15}; W=1 không có trạng thái), `log_rv{k}_med2d`, `r5_2`/`r5_3`, `log_c5_ema5_2`/`_3`; `dow` = ngoại lệ không cửa sổ (không sinh);
  **tổng 163 cột**. (d) TimesFM: giữ kiến trúc LoRA → freeze → XReg, **không** tfm_b0; **calibrate = LoRA FIT + ES chọn epoch** (không có "LoRA
  trước, calibrate sau"); artifact đổi tên `wins/tfm_lora_native.json`, `wins/tfm_lora_xreg.json`, `wins/tfm.json` + metadata (backbone, finetuned,
  finetune_method=LoRA, native, covariates, input_series, context 512, horizon 3, target cumulative y1..y3); log/doc nói "TimesFM-LoRA native".
  (e) **GPU-only hard, không hỏi**: preflight XGBoost kiểm `build_info USE_CUDA` + booster device; mọi vi phạm → `checker_log.hard_fail`.
  (f) **TEST một lần ép trong code**: `final/TEST_SENTINEL.json` (config_hash, checksum data, champion, sha wins) ghi trước khi chạm TEST; lần hai →
  dừng, chỉ `--force-test-rerun` (recovery) vượt và ghi WARN. (g) **Checker không tương tác**: `experiments/<run>/checker_log.jsonl`
  (PASS/INFO/WARN/ERROR); ERROR chặn run tới khi sửa, WARN/INFO ghi rồi tiếp tục; `scripts/checker_record.py`.
- 2026-09-03 (user, vòng expanded-data): S0_m = B0* ∪ F_old_m từ `experiments/15d/wins/<m>.json` (lgbm 14, xgb 11, cat 5, xgbrf 12, lstm 23,
  autots_wr 21, autots_mr 8 cột ext; B0* 72 chung); AutoTS mỗi nhánh kế thừa bộ thắng của nhánh; TimesFM S0 = ∅. Pipeline lgbm/xgb/cat/xgbrf/lstm
  giữ nguyên: calibrate trên S0_m trên data mới → add-one Candidate_m → prune PI chỉ cột mới → confirmation raw vs pruned → win → champion.
  TimesFM = LoRA per fold (FIT học, ES chọn epoch, VAL không thấy) → freeze → XReg add-one trên cùng adapter → tfm-final = {LoRA + XReg(F_best)}
  nếu > +ε_TFM so với native, ngược lại native. Fold-parallel tổng quát (`fold_parallel.py`, `P0_FOLD_WORKERS`/`fold_workers`), candidate tuần tự.
  Không vẽ trong training → `visualize` hậu kỳ. `experiments/**` không bao giờ bị gitignore, nhị phân đi LFS, không commit checkpoint TimesFM gốc.
  Split data đầy đủ `rolling_from_end` neo cuối data, checksum `data/data_checksums_full.json`. LoRA tự chứa (`lora.py`): r=8, α=16, dropout 0,
  AdamW lr 1e-4 wd 0.01, fp32, target `attn.qkv_proj/attn.out/ff0/ff1` × 20 = 80 nn.Linear, 2.048.000 tham số; loss MSE trên ŷ_h = cumsum(r̂);
  `train_forward` tái hiện `compiled_decode` (canary local bit-exact). Chi tiết: `docs/reference/audit_timesfm_lora.md`.
- 2026-09-01 (user chốt): "training chỉ GPU" áp cho CẢ xreg của TimesFM: `jax[cuda12]==0.11.1` + `xreg_force_on_cpu=False`, BẮT BUỘC
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`. `create_covariate_matrix` numpy/sklearn CPU (giới hạn đã ghi). Bằng chứng `scripts/canary_xreg_gpu.py`.
- 2026-08-31 (seed/ε): `calib_seed = 8586` CHỈ ES; `eval_seeds = 8587/8588/8589` đo ε (`noise_cell = 100·std/mean`, `ε = max(0.005, RMS 15 ô)`) và
  confirmation; `selection_seed = 8587` cho MỌI bước selection. AutoTS: WR/MR = probe → `autots-search` framework riêng từng bộ → AutoTS-final; không union.
- 2026-08-29: B0 `TargetTransform` bug → `src/p0/transform.py`; XGB-RF = `XGBRegressor(n_estimators=1, num_parallel_tree=N)`; TimesFM/AutoTS
  trả log-return (`is_logret=True`); prune PI cùng định nghĩa cho mọi loại input.
- 2026-08-28 (rev 8/9/9b): B0* xuất phát chung (vòng 15 ngày); prune PI; confirmation 3 seed mean RMSE từng ô; champion `> +ε_champion`;
  ensemble = champion + model có MedianGain vs E0 > 0 (equal / 1/MSE), < 2 thành viên → không ensemble. ExtraTrees → XGB-RF.
- 2026-08-27 / 08-24: metric trên giá; 39 candidate §2.3 (nay lịch sử); mỗi model feature set riêng; lọc B0 §1.4 một lần → B0* (= R4, 72 cột);
  point-only objective; OHLCV-only; B0 frozen.

## Experiment Findings — VÒNG 15 NGÀY (LỊCH SỬ, đã xong 2026-09-01; artifact `experiments/15d/`)

(dataset `btc_1min_15d_2026-01-18_02-02`, Vast RTX 3090; VAL = 5 fold × 1 ngày, TEST = 02-01→02-02, 2.728 origin; ~27 h máy)

- **Tín hiệu ~0**: trên VAL chỉ xgbrf > E0 (+0.0323 pp); cat −0.0017 · xgb −0.0194 · lgbm −0.0270 · lstm −0.5291 · TimesFM-final (native zero-shot)
  −1.9958 · AutoTS-final −2.0578 ⇒ champion = **xgbrf**, không ensemble. Hợp lệ, không phải lỗi pipeline.
- **TEST Gain vs E0 (pp, h1/h2/h3)**: lgbm +0.247/+0.108/+0.034 · lstm +0.156/+0.095/+0.457 · b0_306 +0.233/−0.067/−0.023 · b0_star +0.149/+0.063/−0.065 ·
  cat +0.111/−0.119/−0.116 · xgbrf +0.088/−0.040/−0.142 · xgb +0.086/−0.010/−1.075 · tfm −1.367/−1.840/−2.914 · autots −2.037/−2.376/−2.853.
  E0 RMSE TEST = 87,25 / 121,31 / 150,44 USD; dir-acc ≈ 0,49–0,52; r ≈ 0,05–0,07. Champion VAL (xgbrf) không phải tốt nhất TEST (chênh ≤ 0,5 pp = nhiễu).
- **ε chi phối KEEP/DROP**: ε xgbrf 0,020 · autots_mr 0,005 · lgbm 0,097 · cat 0,091 · xgb 0,282 · lstm 0,404 · autots_wr 0,681 · TimesFM 0,005 · AutoTS-final 1,166.
  lgbm/xgb/cat/lstm/autots_wr 39 KEEP/0 DROP; xgbrf 31/8; autots_mr 8/31; tfm_b0 & tfm_ext 0/39. Prune PI là bước lọc thật: lgbm 14/40 · xgb 11/40 ·
  cat 5/40 · xgbrf 12/32 · lstm 23/40 · autots_wr 21/40 · autots_mr 5/8 (chọn unprune). Các bộ F_old_m này chính là `locked_ext` của S0_m vòng mới.
- **TimesFM zero-shot**: 72 covariate B0* làm hỏng (−17,7 pp, xreg β nhiễu); native −1,996 pp → TimesFM-final zero-shot = native (B0 = [], ext = []) —
  lý do vòng mới không có nhánh B0* cho TimesFM.
- Không có dấu hiệu leakage (không ô nào > 1 pp dương). Figure vòng này ở `experiments/15d/summary/`.

## Data / Implementation Blockers

- **Data đầy đủ chưa có trên đĩa**: `configs/p0_full.json` trỏ `data/BTC_hf_1min_full.csv` + `data/BTC_lf_5min_full.csv` (user đặt file đúng tên
  hoặc sửa path), checksum `data/data_checksums_full.json` ghi bằng `check-data --write-checksums`. Manifest cũ: 289.320 bar 2026-01-18 → 08-07.
  LF 5' hiện có chỉ đến 03-26 → cần LF mới phủ toàn bộ HF (CLI dừng nếu không phủ).
- Snapshot 15 ngày vẫn ở `data/BTC_hf_1min.csv` + `data/BTC_lf_5min.csv` (config `configs/p0_15d.json`) — lịch sử / `lock-s0 --data-config` xem trước.
- Chi phí vòng mới CHƯA đo: **163 candidate** × 7 model + TimesFM XReg 163 pass × 5 fold × VAL 3 ngày; LoRA 7 adapter/fold × 5 fold; 5 worker
  LoRA cùng lúc chỉ đủ VRAM cho batch ≤ ~35 (audit §7). `scripts/canary_lora.py` đo trước khi cam kết ETA. `short_candidates` trong config cho phép
  giới hạn pool (ghi rõ khi dùng; không phải quyết định mặc định).
- Local: timesfm/autots/jax/peft KHÔNG cài; test TimesFM-LoRA dùng stub + canary local trên sdist random-init. torch 2.11+cu128, RTX 3050 Ti.
- `experiments/15d/wins/*.npz` (27 file) commit dạng blob thường trước khi có LFS — giữ nguyên; file mới đi LFS.
- Windows: `PYTHONPATH` tách bằng `;`; console cp1252 → `run.py` tự reconfigure UTF-8; script ad-hoc cần `PYTHONUTF8=1`.

## Pitfalls

- B0: Huber alpha 0.9 trong z-space; `timestamp_indices` chỉ check `t < end` → harness trừ 3'; Vast dùng LightGBM build CUDA (`device_type="cuda"`,
  bootstrap tự resolve và ghi vào config); GPU histogram không bit-exact.
- TimesFM 2.0.2 (audit LoRA): `decode()` bọc `no_grad` → training đi `train_forward`; RMSNorm `scale` init 0 nên model random-init "chết" (chỉ có ý
  nghĩa với checkpoint thật); inject LoRA SAU `load_checkpoint`, không gọi lại `load_checkpoint`; `torch_compile=False`; kênh 0 = mean head. Với
  k ≥ 1 covariate model nhận phần dư OLS (lệch train/serve vốn có). jax `PREALLOCATE=false` bắt buộc khi torch train cùng GPU.
- AutoTS 1.0.4: bug `sklearn.py:3337` → `fit_data(df)` không truyền regressor rồi tự gán `regressor_train` (MR); `max_windows` mặc định 5000 cắt
  FIT; nhánh xgboost không tự set seed; backend LightGBM của AutoTS theo `models.lgbm.device_type` đã resolve.
- C_short: `log_rv{k}_rv60`, `rsi1`, `bb_pctb_2`… có NaN khi r1 = 0 (≈ 3–4 % bar) — tree nhận NaN native, LSTM điền 0 sau chuẩn hoá (đã có).
- pandas 3.0: CoW; `'min'` thay `'T'`. Target h = 2, 3 chồng lấp → per-bar không iid. Lag-1 autocorr ≈ −0.06 → tín hiệu 0.1–0.2 pp; Gain vài pp = nghi leakage.
- `--smoke`/`--allow-cpu` chỉ với `dataset_label` `synthetic*`. LightGBM/CatBoost predict luôn CPU (đặc tính thư viện).
- Fold-parallel: worker spawn nạp lại store + model qua `cli.model_for` (GPU); run cần predictor sống (prune PI, filter-b0) chạy tuần tự.

## Important Files

- Repo GitHub (private): https://github.com/tson295/P0_forecasting — branch `main`; raw CSV không push; experiments/** tracked (LFS nhị phân).
- `configs/p0_full.json` (vòng mới) · `configs/p0_15d.json` (lịch sử). `experiments/full/s0/` (S0_m, Candidate_m, collisions.json, short_pool.json);
  `experiments/full/checker_log.jsonl` (finding); `experiments/full/final/TEST_SENTINEL.json` (sinh khi final).
- `src/p0/`: `features_short.py` (C_short 163), `s0.py`, `checker_log.py`, `fold_parallel.py`, `lora.py`, `models_tfm.py` (`TimesFMLoRAModel`),
  `visualize.py`, `cli.py` (check-data / lock-s0 / loop / tfm-final / autots-search / ensemble / final [--force-test-rerun] / visualize), `split.py`.
- `scripts/`: `checker_record.py` (checker ghi finding), `canary_lora.py`, `vast_canary.py`, `canary_xreg_gpu.py`, `vast_bootstrap.sh`.
- `docs/RESEARCH_PLAN.md` rev 10.1 · `docs/reference/audit_timesfm_lora.md`, `audit_timesfm.md`, `audit_autots.md`.
- `experiments/15d/` — toàn bộ vòng 15 ngày. `Baseline_LGBM.py` — B0 frozen (deny Edit/Write).

## Open Questions

- Thời gian/VRAM thật của LoRA và XReg 163 candidate trên RTX 3090 — canary sau unlock; có giới hạn `short_candidates` hay không — user quyết.
- Tên file data đầy đủ (`*_full.csv`) và LF 5' phủ đủ — user cung cấp.
- Có mở cross-asset (ETH/SOL/XRP) làm feature ở vòng sau hay không — user quyết.
