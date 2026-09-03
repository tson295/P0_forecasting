# MEMORY — trạng thái (update/replace, không append mâu thuẫn)

PHASE: **MIGRATION EXPANDED-DATA XONG (code/config/doc) — chờ checker + user unlock**, 2026-09-03
TRAINING: LOCKED

## Current Task

Vòng mới **expanded-data** (quyết định user 2026-09-03). Codebase đã được sửa để sẵn sàng: S0_m khoá từ artifact 15 ngày,
C_short (97 candidate ngắn ≤ 15'), Candidate_m, TimesFM-LoRA → freeze → XReg search, fold-parallel tổng quát, không vẽ trong
training, `experiments/**` không bị gitignore (LFS cho .npz/.pt), split rolling neo vào cuối data thật. **Chưa chạy training nào,
chưa chạm TEST.** Vòng 15 ngày (2026-08-31 → 09-01) hoàn tất, artifact + kết quả nằm ở `experiments/15d/` (lịch sử, không sửa).

## Exact Next Step

1. Đặt data đầy đủ vào `data/BTC_hf_1min_full.csv` + `data/BTC_lf_5min_full.csv` (LF phải phủ toàn bộ HF; nhiều S0_m có cột 5').
   `python run.py check-data --config configs/p0_full.json --write-checksums` → in fold rolling (5 × VAL 3 ngày, FIT 40 + ES 5, TEST 30 ngày cuối, cần ≥ 90 ngày).
2. `python run.py lock-s0 --config configs/p0_full.json` → `experiments/full/s0/` (S0_m, Candidate_m, collisions.json) trên data mới.
   (Đã chạy thử với `--data-config configs/p0_15d.json` khi migration: 0 cặp trùng giá trị, 5 cặp near chỉ báo; báo cáo ở `experiments/full/s0/`.)
3. Trên Vast: `bash scripts/vast_bootstrap.sh` → `scripts/vast_canary.py` + `scripts/canary_xreg_gpu.py` → `pytest -q -x`; sau unlock: `scripts/canary_lora.py`
   (1 fold × 1 epoch × 64 origin, adapter thư mục tạm, đi qua gate) để đo thời gian/VRAM trước khi cam kết ETA. Chú ý: `loop` từ chối nếu
   `s0/candidates_<m>.json` được audit trên dataset khác config → phải chạy lại `lock-s0` trên data thật (không --data-config).
4. **User unlock rõ ràng** → `docs/VAST_SESSION_PROMPT.md` (thứ tự plan §8: loop lgbm → xgb → cat → tfm → tfm-final → xgbrf → autots_wr → autots_mr → autots-search → lstm → ensemble → final → visualize).

## Decisions (mới nhất trước)

- 2026-09-03 (user, vòng expanded-data — thay thế mọi quyết định cũ về điểm xuất phát/candidate/TimesFM):
  (a) **S0_m = B0* ∪ F_old_m KHOÁ** từ `experiments/15d/wins/<m>.json` (lgbm 14, xgb 11, cat 5, xgbrf 12, lstm 23, autots_wr 21, autots_mr 8 cột ext;
  B0* 72 cột chung); cột khoá không phải candidate, không bị prune PI, không thể bỏ. AutoTS: mỗi nhánh probe kế thừa đúng bộ thắng của nhánh đó.
  TimesFM: S0 = ∅ (TimesFM-final cũ = native). (b) **Candidate = CHỈ C_short** (`features_short.py`): lưới {1,2,3,4,5,8,10,15} làm dày 14 họ
  A–O (bỏ họ H Keltner ngắn vì ≡ log_atr_k_c + const, corr ≥ 0.999999 trên data thật) = **97 cột**; candidate cũ (KEEP/DROP) không quay lại;
  Candidate_m = C_short \ overlap(C_short, S0_m) (trùng tên hoặc trùng giá trị, kiểm bằng số trong `s0.collision_audit`; cùng indicator khác lag KHÔNG
  phải trùng; tương quan cao chỉ báo). Lưu `experiments/<run>/s0/`. (c) **Pipeline giữ nguyên** cho lgbm/xgb/cat/xgbrf/lstm: calibrate trên S0_m
  trên data mới (số vòng/epoch, ε mới — KHÔNG kế thừa số cũ) → add-one Candidate_m → prune PI CHỈ cột mới → confirmation raw vs pruned → win_m → champion.
  (d) **TimesFM = LoRA per fold** (FIT học, ES chọn epoch, VAL không thấy) → freeze adapter → XReg add-one trên CÙNG adapter (thêm candidate = fit lại
  xreg, không động trọng số); native LoRA lưu `wins/tfm_native.json`; `tfm-final` = {LoRA + XReg(win)} nếu MedianGain vs native > +ε_TFM, ngược lại native.
  Không có nhánh `tfm_b0`/`tfm_ext` nữa. Adapter theo 3 vai trò seed (calib → epoch ES; eval seeds → ε; selection_seed → adapter FROZEN duy nhất cho
  add-one + prune; confirmation ES bật 3 seed). (e) **Fold-parallel tổng quát** (`fold_parallel.py`, `P0_FOLD_WORKERS`/`fold_workers`): kết quả y hệt
  tuần tự, thứ tự fold tất định, không CPU fallback (hết VRAM → fail rõ, giảm worker); candidate vẫn tuần tự. (f) **Không vẽ trong training**:
  mọi artifact lưu (wins/*.npz, final/*.npz + index.json, champion_log, prune_pi, calib, LoRA .pt) → `python run.py visualize` dựng lại sau.
  (g) **experiments/** không bao giờ bị gitignore**; .npz/.safetensors/.pt/.pth dưới experiments/ đi qua Git LFS; checkpoint TimesFM gốc không commit
  (chỉ model id + revision). (h) Split data đầy đủ: `split.rolling_from_end` neo vào cuối data thật (không hard-code ngày), checksum riêng
  `data/data_checksums_full.json`. (i) LoRA tự chứa (`lora.py`, không thêm peft/transformers): r=8, α=16, dropout 0, AdamW lr 1e-4 wd 0.01, fp32,
  target `attn.qkv_proj/attn.out/ff0/ff1` × 20 layer = 80 nn.Linear, 2.048.000 tham số; loss = MSE trên ŷ_h = cumsum(r̂) vs y_h (mean head, patch cuối);
  `train_forward` tái hiện đúng `compiled_decode` (canary local bit-exact). Chi tiết: `docs/reference/audit_timesfm_lora.md`.
- 2026-09-01 (user chốt): **invariant "training chỉ GPU" áp dụng cho CẢ bước xreg của TimesFM**: `jax[cuda12]==0.11.1` + `xreg_force_on_cpu=False`,
  BẮT BUỘC `XLA_PYTHON_CLIENT_PREALLOCATE=false`. `create_covariate_matrix` trong timesfm là numpy/sklearn CPU (không có tuỳ chọn GPU) — giới hạn đã ghi.
  Bằng chứng `scripts/canary_xreg_gpu.py`; 2,9× nhanh hơn; lệch float32 5,2e-06 so với CPU.
- 2026-08-31 (seed/ε, user chốt): ba vai trò seed tách bạch. `calib_seed = 8586` CHỈ cho ES lấy số vòng/epoch. `eval_seeds = 8587/8588/8589` đo ε
  (`noise_cell = 100·std/mean` từng ô, `ε = max(0.005, RMS 15 ô)`) và confirmation 3 seed. `selection_seed = 8587` cho MỌI bước selection.
- 2026-08-31 (AutoTS): WR/MR = probe (mỗi cái add-one → prune → confirmation → F_WR_best / F_MR_best); `autots-search` chạy framework AutoTS riêng cho
  từng bộ (template GPU khai báo, `max_generations=0`, search chỉ trên FIT+ES) → AutoTS-final. Không có stage union.
- 2026-08-31 (agent): 4 agent — checker, researcher, analyst, infra.
- 2026-08-29: B0 `TargetTransform` có bug nhân in-place → harness dùng `src/p0/transform.py` (cùng công thức); XGB-RF = `XGBRegressor(n_estimators=1,
  num_parallel_tree=N)`; TimesFM/AutoTS trả log-return trực tiếp (`is_logret=True`); prune PI cùng định nghĩa cho mọi loại input.
- 2026-08-28 (rev 8/9/9b): B0* điểm xuất phát chung (vòng 15 ngày); prune PI, confirmation 3 seed mean RMSE từng ô; champion `> +ε_champion`; ensemble =
  champion + model có MedianGain vs E0 > 0 (equal / 1/MSE), < 2 thành viên → không ensemble. ExtraTrees → XGB-RF. Cờ + = > 0 ở ≥ 2/3 horizon.
- 2026-08-27: metric trên giá; 39 candidate §2.3 cũ; mỗi model feature set riêng; lọc B0 §1.4 một lần → B0* (= R4, 72 cột, đã xong).
- 2026-08-24: point-only objective; OHLCV-only; B0 frozen; Gain/MedianGain/WinRate/P10/Worst + RMSE/MAE/r/dir-acc.

## Experiment Findings — VÒNG 15 NGÀY (LỊCH SỬ, đã xong 2026-09-01; artifact `experiments/15d/`)

(dataset `btc_1min_15d_2026-01-18_02-02`, Vast RTX 3090; VAL = 5 fold × 1 ngày, TEST = 02-01→02-02, 2.728 origin; ~27 h máy)

- **Tín hiệu ~0**: trên VAL chỉ xgbrf > E0 (+0.0323 pp); cat −0.0017 · xgb −0.0194 · lgbm −0.0270 · lstm −0.5291 · TimesFM-final (native zero-shot)
  −1.9958 · AutoTS-final −2.0578 ⇒ champion = **xgbrf**, không ensemble. Hợp lệ, không phải lỗi pipeline.
- **TEST Gain vs E0 (pp, h1/h2/h3)**: lgbm +0.247/+0.108/+0.034 · lstm +0.156/+0.095/+0.457 · b0_306 +0.233/−0.067/−0.023 · b0_star +0.149/+0.063/−0.065 ·
  cat +0.111/−0.119/−0.116 · xgbrf +0.088/−0.040/−0.142 · xgb +0.086/−0.010/−1.075 · tfm −1.367/−1.840/−2.914 · autots −2.037/−2.376/−2.853.
  E0 RMSE TEST = 87,25 / 121,31 / 150,44 USD; dir-acc ≈ 0,49–0,52; r ≈ 0,05–0,07. Champion VAL (xgbrf) không phải tốt nhất TEST (chênh ≤ 0,5 pp = nhiễu).
- **ε chi phối KEEP/DROP**: ε xgbrf 0,020 · autots_mr 0,005 · lgbm 0,097 · cat 0,091 · xgb 0,282 · lstm 0,404 · autots_wr 0,681 · TimesFM 0,005 · AutoTS-final 1,166.
  lgbm/xgb/cat/lstm/autots_wr 39 KEEP/0 DROP; xgbrf 31/8; autots_mr 8/31; tfm_b0 & tfm_ext 0/39. Prune PI là bước lọc thật: lgbm 14/40 · xgb 11/40 ·
  cat 5/40 · xgbrf 12/32 · lstm 23/40 · autots_wr 21/40 · autots_mr 5/8 (chọn unprune). Các bộ F_old_m này chính là phần khoá của S0_m vòng mới.
- **TimesFM zero-shot**: 72 covariate B0* làm hỏng (−17,7 pp, xreg β nhiễu, sai số tăng tuyến tính theo h); native −1,996 pp → TimesFM-final = native.
  Vòng mới thay bằng LoRA + XReg trên cùng adapter (quyết định 2026-09-03).
- Không có dấu hiệu leakage (không ô nào > 1 pp dương). Figure §7.3 của vòng này ở `experiments/15d/summary/`.

## Data / Implementation Blockers

- **Data đầy đủ chưa có trên đĩa**: config `configs/p0_full.json` trỏ `data/BTC_hf_1min_full.csv` + `data/BTC_lf_5min_full.csv` (tên file do migration
  đặt — user đặt file đúng tên hoặc sửa path), checksum `data/data_checksums_full.json` ghi bằng `check-data --write-checksums` khi có data. Manifest cũ:
  289.320 bar 2026-01-18 16:15 → 08-07 14:14. LF 5' hiện có chỉ đến 03-26 → cần LF mới phủ toàn bộ HF (CLI từ chối nếu không phủ).
- Snapshot 15 ngày vẫn ở `data/BTC_hf_1min.csv` + `data/BTC_lf_5min.csv` (checksum `data/data_checksums.json`, config `configs/p0_15d.json`) — chỉ để
  lịch sử / chạy `lock-s0 --data-config` / visualize lại `experiments/15d`.
- Chi phí vòng mới CHƯA đo: 97 candidate × 7 model tree/LSTM/AutoTS + TimesFM XReg 97 pass × 5 fold × VAL 3 ngày (4.320 origin) — lớn hơn vòng 15 ngày
  nhiều lần; LoRA 7 adapter/fold (1 calib + 3 eval + 3 confirmation) × 5 fold; VRAM khi 5 worker LoRA cùng lúc chỉ đủ batch ≤ ~35 (audit §7) → cân nhắc
  `fold_workers` nhỏ hơn cho `tfm` hoặc train tuần tự. Cần canary thời gian thật trước khi cam kết ETA. Có thể trim pool bằng `short_candidates` trong config.
- Local: timesfm/autots/jax/peft KHÔNG cài; test TimesFM-LoRA dùng stub + canary local trên sdist random-init (scratchpad). torch 2.11+cu128, RTX 3050 Ti.
- `experiments/15d/wins/*.npz` (27 file) đã commit dạng blob thường trước khi có LFS — giữ nguyên, không rewrite history; file mới đi LFS.
- Windows: `PYTHONPATH` tách bằng `;`; console cp1252 → `run.py` tự reconfigure UTF-8; script ad-hoc cần `PYTHONUTF8=1`.

## Pitfalls

- B0: Huber alpha 0.9 trong z-space; `timestamp_indices` chỉ check `t < end` → harness trừ 3'; `device_type="gpu"` = build OpenCL (Vast dùng build CUDA →
  `device_type="cuda"`, bootstrap tự resolve và ghi vào config); GPU histogram không bit-exact.
- TimesFM 2.0.2 (audit LoRA): `decode()` bọc `no_grad` → training phải đi `train_forward` (tái hiện normalize/revin/running stats/flip); RMSNorm `scale`
  init 0 nên model random-init "chết" (chỉ tokenizer) — chỉ có ý nghĩa với checkpoint thật; inject LoRA SAU `load_checkpoint`, không gọi lại
  `load_checkpoint` trên module đã inject; `torch_compile=False`; kênh 0 = mean head; `point_forecast` = q50. Với k ≥ 1 covariate, model nhận phần dư OLS
  (lệch train/serve vốn có của "train một lần, freeze"). jax `PREALLOCATE=false` bắt buộc khi torch train cùng GPU.
- AutoTS 1.0.4: bug `sklearn.py:3337` → `fit_data(df)` không truyền regressor rồi tự gán `regressor_train` (MR); `max_windows` mặc định 5000 cắt FIT;
  nhánh xgboost không tự set seed; backend LightGBM của AutoTS phải theo `models.lgbm.device_type` đã resolve.
- pandas 3.0: CoW; `'min'` thay `'T'`. Target h = 2, 3 chồng lấp → per-bar không iid.
- Lag-1 autocorr 1-min ≈ −0.06 → tín hiệu cỡ 0.1–0.2 pp RMSE ở h=1; Gain vài pp = nghi leakage; forecast "phẳng" là bình thường.
- `--smoke`/`--allow-cpu` chỉ với `dataset_label` `synthetic*`. LightGBM/CatBoost predict luôn CPU (đặc tính thư viện).
- Fold-parallel: worker spawn nạp lại store + model qua `cli.model_for` (GPU); run cần predictor sống (prune PI, filter-b0) chạy tuần tự; confirmation
  đo latency trong worker fold đầu. Không parallel candidate.

## Important Files

- Repo GitHub (private): https://github.com/tson295/P0_forecasting — branch `main`; raw CSV không push; artifact experiments/** tracked (LFS cho nhị phân).
- `configs/p0_full.json` (vòng mới: `experiments/full`, `prev_run_dir: experiments/15d`, split rolling, `fold_workers: 5`) · `configs/p0_15d.json` (lịch sử).
- `src/p0/`: `features_short.py` (C_short), `s0.py` (S0_m/collision/Candidate_m), `fold_parallel.py`, `lora.py`, `models_tfm.py` (`TimesFMLoRAModel`),
  `visualize.py` (hậu kỳ), `cli.py` (lock-s0 / loop / tfm-final / autots-search / ensemble / final / visualize), `split.py` (`RollingSpec`).
- `docs/RESEARCH_PLAN.md` rev 10 (2026-09-03) · `docs/reference/audit_timesfm_lora.md` (mới) · `audit_timesfm.md`, `audit_autots.md` (2026-08-29/31).
- `experiments/15d/` — toàn bộ vòng 15 ngày (log, wins, keepdrop, prune, champion_log, summary/figure, predictions VAL, val_paths).
- `experiments/full/s0/` — S0_m + Candidate_m + collisions.json (audit bằng số trên data 15 ngày; chạy lại `lock-s0` khi có data mới).
- `Baseline_LGBM.py` — B0 frozen (deny Edit/Write). `scripts/export_val_predictions.py`, `scripts/fig_val_paths_all_models.py` — helper cho vòng 15 ngày.

## Open Questions

- Thời gian/VRAM thật của LoRA trên RTX 3090 (step/epoch, 5 worker) — canary trước khi cam kết ETA; có trim `short_candidates` hay không — user quyết.
- Tên file data đầy đủ (`*_full.csv`) và LF 5' phủ đủ — user cung cấp.
- Có mở cross-asset (ETH/SOL/XRP) làm feature ở vòng sau hay không — user quyết.
