# MEMORY — trạng thái (update/replace, không append mâu thuẫn)

PHASE: **PHA VẬN HÀNH — data 2 năm ĐÃ TRONG REPO (Git LFS), scheduler 2 GPU đối xứng + orchestrate + champion replay,
agent chuyển sang checker/run-monitor/infra (analyst hậu-run, researcher dormant) — chờ preflight Vast + user unlock**, 2026-09-04d
TRAINING: LOCKED

## Current Task

Vòng **expanded-data** trên **data 2 năm thật** (quyết định user 2026-09-04b): `data/BTC_1m_2y.csv` (1.051.201 bar 60 s, 2024-09-03 16:29 →
2026-09-03 16:29 UTC, sha256 `559ce040…f097`) + LF 5' dẫn xuất tất định `data/BTC_5m_2y.csv` (210.239 bar, sha256 `0e5fb9ad…f2fef`).
**Từ 2026-09-04d cả hai file NẰM TRONG REPO qua Git LFS** → `git clone` + `git lfs pull` là đủ để `check-data`, không scp, không cần `derive-lf`.
Split `rolling_spread`: 5 VAL 3 ngày rải đều 2025-01-07 → 2026-08-01, FIT 120 ngày rolling + ES 5 ngày, TEST 30 ngày cuối (08-04 16:30 → 09-03 16:30).
`lock-s0` đã chạy trên data thật: Candidate_m = 163 cho mọi model (0 overlap với S0_m). **Chưa chạy training nào, chưa chạm TEST.**
Pass 2026-09-04c là **THỰC THI/SCHEDULING** cho máy **2 × RTX 5000 Ada 32 GB**: scheduler GPU đối xứng (`src/p0/gpu.py` + `scheduler.py`,
`fold_parallel.py` thành adapter), `orchestrate` chạy DAG nhánh model song song, champion HOÃN → `champion-replay` thứ tự cố định,
TimesFM đổi tên hai HỆ THỐNG HOÀN CHỈNH (A = `tfm_lora_baseline` feature-free vs B = `tfm_lora_xreg` = LoRA + XReg(F_win)).
**Methodology không đổi một dòng** so với `afca1c8`/`83b004e`: target, feature, S0/Candidate_m, thứ tự candidate, KEEP/DROP, PI,
confirmation, seed, ε, hyperparameter, split, TEST, metric, luật champion/ensemble. `config_hash` của `p0_full.json` vẫn `7169b3d4ea38`.

## Exact Next Step

1. Trên Vast: `git clone` → `git lfs install && git lfs pull` (đã có CẢ hai CSV; `git lfs ls-files | grep BTC_` phải thấy đủ,
   file ~101,8 MB + ~21,1 MB — nếu ~130 byte là quên `lfs pull`), rồi:
   `bash scripts/vast_bootstrap.sh` → `scripts/vast_canary.py` + `scripts/canary_xreg_gpu.py` →
   `python run.py check-data --config configs/p0_full.json` (verify anchor `data/data_checksums_2y.json`, in 5 fold + final như dưới) →
   `python run.py lock-s0 --config configs/p0_full.json` (phải ra 163/model, 0 overlap; audit label = `btc_1min_2y_2024-09-03_2026-09-03`) →
   `pytest -q -x` → `python scripts/checker_record.py --exp experiments/full --blocking` sạch ERROR → agent `checker` (không tương tác).
   Trên máy 2 GPU: `export P0_GPU_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false` (KHÔNG đặt `P0_FOLD_WORKERS`), rồi
   `python run.py gpu-probe --config configs/p0_full.json` — worker 0 → GPU vật lý 0, worker 1 → GPU vật lý 1, **UUID phải khác nhau**,
   mọi backend đã cài phải chạy được GPU trong worker (kết quả → `experiments/full/gpu_probe.json`). Lỗi GPU ở bất kỳ đâu = exit 3 +
   ERROR `ref=USER_DECISION_REQUIRED` → **hỏi user**, không CPU fallback.
2. **User unlock rõ ràng** → `scripts/canary_lora.py` (1 fold × 1 epoch × 64 origin, đo thời gian/VRAM; FIT 172.7k cửa sổ/fold) →
   `python run.py orchestrate --config configs/p0_full.json` (DAG nhánh song song → champion-replay → ensemble; KHÔNG chạm TEST)
   hoặc từng bước như `docs/VAST_SESSION_PROMPT.md` → `python run.py final` (TEST một lần) → `visualize`.

## Split đã resolve trên data thật (check-data 2026-09-04; origin = eligible B0, first 2024-09-04 03:00 UTC)

| Fold | FIT (120 ngày) | ES (5 ngày − purge) | VAL (3 ngày) | n FIT / ES / VAL |
|---|---|---|---|---|
| 1 | 2024-09-04 03:00 → 2025-01-02 03:00 | → 01-07 02:00 | 2025-01-07 03:00 → 01-10 03:00 | 172.753 / 7.121 / 4.317 |
| 2 | 2025-01-25 00:22 → 05-25 00:22 | → 05-29 23:22 | 2025-05-30 00:22 → 06-02 00:22 | 172.727 / 7.137 / 4.309 |
| 3 | 2025-06-16 21:45 → 10-14 21:45 | → 10-19 20:45 | 2025-10-19 21:45 → 10-22 21:45 | 172.409 / 7.137 / 4.317 |
| 4 | 2025-11-06 19:08 → 2026-03-06 19:08 | → 03-11 18:08 | 2026-03-11 19:08 → 03-14 19:08 | 172.554 / 7.137 / 4.317 |
| 5 | 2026-03-29 16:30 → 07-27 16:30 | → 08-01 15:30 | 2026-08-01 16:30 → 08-04 16:30 | 172.703 / 7.109 / 4.300 |
| final | 2026-04-01 16:30 → 07-30 16:30 | → 08-04 15:30 | TEST 2026-08-04 16:30 → 09-03 16:30 | 172.703 / 7.085 / **42.918** |

Lý do rải đều (EDA user): năm 1 +94 %, năm 2 −27,9 %, max drawdown −54 %, vol 1' 3,7 → 9,7 bp theo tháng; hướng target cân bằng (≈ 50 %)
→ giá trị của 2 năm là ĐA DẠNG REGIME, 5 ô VAL phải lấy mẫu regime khác nhau. FIT 120 ngày (≈ 172,8k bar) thay 40 ngày; KHÔNG expanding.

## Decisions (mới nhất trước)

- 2026-09-04d (user, pass VẬN HÀNH — không đổi khoa học): (a) **hai CSV canonical vào REPO qua Git LFS**
  (`.gitignore`: `data/*.csv` + `!data/BTC_1m_2y.csv` + `!data/BTC_5m_2y.csv`; `.gitattributes` LFS ĐÚNG hai file đó) → quy trình
  chuẩn `git clone` → `git lfs pull` → `check-data`; `derive-lf` ở lại làm công cụ tái lập/kiểm chứng. (b) Audit
  `git check-ignore`: KHÔNG artifact nào dưới `experiments/**` bị ignore (có test). (c) **TimesFM khoá cách gọi**: mọi cấu hình
  feature được chấm dưới dạng HỆ THỐNG HOÀN CHỈNH `TimesFM-LoRA + XReg(F)`; confirmation = {LoRA+XReg(F_raw)} vs
  {LoRA+XReg(F_pruned)} trên CÙNG adapter (code assert danh tính adapter) → F_win; rồi A (LoRA feature-free) vs B (LoRA+XReg(F_win))
  → TFM-final; cấm gọi "XReg vs XReg"/"XReg vs LoRA" (có test quét code + doc). (d) **TFM-final và AutoTS-final LƯU rồi CHỜ**
  champion replay (defer_champion). (e) `max_branches: 4` nhưng `gpu_slots_per_device: 1` → vẫn tối đa 2 task nặng.
  (f) **Sự cố TÀI NGUYÊN GPU = ngoại lệ tương tác DUY NHẤT**: `checker_log.gpu_stop` (ERROR `ref=USER_DECISION_REQUIRED`, exit 3,
  giữ artifact, không CPU fallback, không đổi tham số) — dùng cho GPU preflight, worker chết/không khởi động, task lỗi mang dấu hiệu
  GPU/OOM, `gpu-probe` UUID trùng hoặc backend đã cài mà không chạy được GPU; vi phạm bất biến khoa học vẫn `hard_fail` im lặng.
  (g) `gpu-probe` mạnh hơn: UUID phân biệt + `backend_probe` (torch/xgboost + booster device/lightgbm/catboost/jax/timesfm) chạy
  TRONG worker đã mask, lưu `experiments/<run>/gpu_probe.json`. (h) **Agent pha vận hành**: thêm `run-monitor` (chỉ đọc),
  `checker` chỉ hai điểm (trước orchestrate, trước final), `analyst` hậu-run, `researcher` dormant, `infra` on-demand.

- 2026-09-04c (user, pass THỰC THI — không đổi khoa học): (a) **TimesFM ngữ nghĩa**: candidate XReg → F_raw → prune PI → F_pruned →
  **confirmation raw vs pruned → F_win** → RỒI MỚI so **hai hệ thống hoàn chỉnh** A = `wins/tfm_lora_baseline.json` (LoRA đã fine-tune,
  0 feature/0 B0*/0 covariate) vs B = `wins/tfm_lora_xreg.json` (CÙNG adapter freeze + XReg(F_win)) → `tfm-final` → `wins/tfm.json`;
  cấm gọi "XReg vs LoRA"; artifact ghi `system` A/B, `feature_set_source`, `lora_adapters` (A và B phải trùng); tên cũ
  `tfm_lora_native.json` vẫn ĐỌC được. `champion_step` chặn cứng `tfm_lora_*`/`autots_wr`/`autots_mr` (CHAMPION_INELIGIBLE).
  (b) **Scheduler 2 GPU đối xứng**: worker = `len(gpu_devices) × gpu_slots_per_device` (config `[0,1] × 1`), mỗi worker là process khoá
  vào MỘT GPU vật lý bằng `CUDA_VISIBLE_DEVICES` đặt trước import CUDA (cơ chế duy nhất mọi backend đều tôn trọng); KHÔNG vai trò ML/DL,
  KHÔNG pin family; task sẵn sàng → GPU rảnh (round-robin giữa nhánh, FIFO trong nhánh); 5 fold rải động, candidate vẫn tuần tự;
  prune PI trọn trong 1 worker (giữ một dòng RNG); parent không chạy CUDA khi scheduler bật (autots bake-off/score cũng thành task);
  không CPU fallback; log `scheduler_log.jsonl`. (c) **`orchestrate`**: DAG nhánh (loop độc lập ‖; tfm-final ← loop tfm;
  autots-search ← autots_wr + autots_mr), `max_branches` mặc định = số worker; `orchestrate_log.jsonl`. (d) **Champion HOÃN**
  (`defer_champion: true`) → `champion-replay` so theo THỨ TỰ CỐ ĐỊNH lgbm→xgb→cat→tfm→xgbrf→autots→lstm, CHỈ đọc artifact
  (`champion_extra` trong wins), không train/inference; `champion_replay.csv/json`. (e) `final` (TEST) vẫn tuần tự, lệnh riêng,
  orchestrate không bao giờ chạm TEST. (f) `gpu-probe` kiểm định tuyến GPU thật. (g) `gpu_devices`/`gpu_slots_per_device`/`max_branches`/
  `defer_champion` KHÔNG vào `config_hash` (chỉ thực thi).

- 2026-09-04b (user, data 2 năm): (a) nguồn canonical `data/BTC_1m_2y.csv` (cột `ts` → alias `timestamp` trong bộ nhớ; có cả hai phải trùng;
  `datetime` kiểm khớp epoch UTC rồi dựng lại); KHÔNG dùng `data/BTC_hf_1min_full.csv`. (b) LF 5' dẫn xuất từ HF bằng `derive-lf`
  (`data.derive_lf_5min`): nhóm (T−4'..T], nhãn T = bar cuối (bội 300 s), open first / high max / low min / close last / volume+amount sum,
  bỏ nhóm thiếu bar (2 nhóm), tất định (cùng byte), as-of `T ≤ t`; sidecar `data/BTC_5m_2y.derivation.json` ghi sha nguồn, `check-data`
  hard-fail nếu LF không dẫn xuất từ HF hiện tại. (c) Anchor mới `data/data_checksums_2y.json` (không ghi đè anchor cũ). (d) Split
  `rolling_spread` (`split.make_rolling_spread`): n_folds 5, fit 120, es 5, val 3, test 30, purge 60; VAL rải đều linspace(earliest, latest)
  làm tròn phút; FIT/ES/VAL mỗi fold `[vs−es−fit, vs−es) / [vs−es, vs−purge) / [vs, vs+val)`; Final FIT 120 + ES 5 trước TEST; không
  expanding, không FIT 365/730 cho từng candidate. Split 15 ngày lịch sử (`rolling_from_end`, `make_folds`) giữ nguyên.
- 2026-09-04 (user, pass hiệu chỉnh — commit `afca1c8`): toàn bộ S0_m khoá (`locked_b0`/`locked_ext`); Candidate_m = C_short \ overlap(S0_m) per
  model, không lọc toàn cục, tương quan cao chỉ báo cáo (bản 2026-09-03 từng ghi sai "user bỏ Keltner vì corr ≥ 0.999999" — đã gỡ); C_short dày
  163 (Keltner, PSAR cửa sổ reset, log_rv_k_med2d, r5_2/3, log_c5_ema5_2/3; dow ngoại lệ); TimesFM calibrate = LoRA FIT + ES, artifact
  `tfm_lora_native`/`tfm_lora_xreg` + metadata, không B0*; GPU-only hard, TEST sentinel một lần, checker không tương tác → `checker_log.jsonl`.
- 2026-09-03 (user, vòng expanded-data): S0_m = B0* ∪ F_old_m từ `experiments/15d/wins/<m>.json` (lgbm 14, xgb 11, cat 5, xgbrf 12, lstm 23,
  autots_wr 21, autots_mr 8; B0* 72); pipeline lgbm/xgb/cat/xgbrf/lstm không đổi; TimesFM LoRA → freeze → XReg; fold-parallel; không vẽ trong
  training; experiments/** tracked + LFS; LoRA tự chứa r=8 α=16, 80 nn.Linear, loss MSE trên ŷ_h. Chi tiết `docs/reference/audit_timesfm_lora.md`.
- 2026-09-01: xreg TimesFM trên jax GPU (`jax[cuda12]==0.11.1`, `PREALLOCATE=false`). 2026-08-31: ba vai trò seed; AutoTS probe → bake-off.
- 2026-08-29 / 08-28 / 08-27 / 08-24: B0 `TargetTransform` bug → `transform.py`; XGB-RF; prune PI + confirmation 3 seed; champion `> +ε`;
  ensemble; metric trên giá; lọc B0 → B0* (R4, 72 cột); point-only; OHLCV-only; B0 frozen.

## Experiment Findings — VÒNG 15 NGÀY (LỊCH SỬ, đã xong 2026-09-01; artifact `experiments/15d/`)

(dataset `btc_1min_15d_2026-01-18_02-02`, Vast RTX 3090; VAL = 5 fold × 1 ngày, TEST = 02-01→02-02, 2.728 origin; ~27 h máy)

- **Tín hiệu ~0**: trên VAL chỉ xgbrf > E0 (+0.0323 pp); cat −0.0017 · xgb −0.0194 · lgbm −0.0270 · lstm −0.5291 · TimesFM-final (native zero-shot)
  −1.9958 · AutoTS-final −2.0578 ⇒ champion = **xgbrf**, không ensemble.
- **TEST Gain vs E0 (pp, h1/h2/h3)**: lgbm +0.247/+0.108/+0.034 · lstm +0.156/+0.095/+0.457 · b0_306 +0.233/−0.067/−0.023 · b0_star +0.149/+0.063/−0.065 ·
  cat +0.111/−0.119/−0.116 · xgbrf +0.088/−0.040/−0.142 · xgb +0.086/−0.010/−1.075 · tfm −1.367/−1.840/−2.914 · autots −2.037/−2.376/−2.853.
  E0 RMSE TEST = 87,25 / 121,31 / 150,44 USD. Champion VAL (xgbrf) không phải tốt nhất TEST (chênh ≤ 0,5 pp = nhiễu).
- **ε chi phối KEEP/DROP**; prune PI là bước lọc thật (lgbm 14/40 · xgb 11/40 · cat 5/40 · xgbrf 12/32 · lstm 23/40 · autots_wr 21/40 · autots_mr 5/8
  chọn unprune) — các bộ F_old_m này là `locked_ext` của S0_m vòng mới.
- **TimesFM zero-shot**: 72 covariate B0* làm hỏng (−17,7 pp); native −1,996 pp → lý do vòng mới không có nhánh B0* cho TimesFM.
- Không có dấu hiệu leakage. Figure vòng này ở `experiments/15d/summary/`.

## Data / Implementation Blockers

- Data 2 năm: HF 1.051.201 bar, 0 gap/dup, OHLC/amount hợp lệ; B0-eligible 1.049.358 origin (warmup 631 bar); LF dẫn xuất 210.239 bar
  (2024-09-03 16:35 → 2026-09-03 16:25, bỏ 2 nhóm thiếu đầu/cuối), phủ HF (lệch cuối 240 s < 300 s). CSV raw/dẫn xuất KHÔNG vào git
  (`data/*.csv` ignore); anchor + sidecar được track.
- Chi phí vòng mới CHƯA đo: FIT 172,7k origin/fold (×3 so với 40 ngày), 163 candidate × 7 model; TimesFM XReg 163 pass × 5 fold × ~4.317 origin
  (~3,5 M lời gọi covariate 1 origin — rất nặng); LoRA 7 adapter/fold × 5 fold trên 172,7k cửa sổ. `scripts/canary_lora.py` đo trước khi cam kết ETA;
  `short_candidates` trong config cho phép giới hạn pool (ghi rõ khi dùng; không phải mặc định). VRAM 5 worker LoRA chỉ đủ batch ≤ ~35 (audit §7).
- `experiments/full/`: `s0/` (data thật), `checker_log.jsonl`; chưa có wins/lora/runs/final/scheduler_log (sinh khi chạy thật).
- Scheduler CHƯA chạy trên 2 GPU thật: mọi kiểm tra 2 worker mới ở mức test (CPU, data tổng hợp) + `gpu-probe` local 1 GPU (RTX 3050 Ti). Phải chạy `gpu-probe` trên máy 2 × RTX 5000 Ada trước khi training.
- Snapshot 15 ngày vẫn ở `data/BTC_hf_1min.csv` + `data/BTC_lf_5min.csv` (config `configs/p0_15d.json`, anchor `data/data_checksums.json`) — lịch sử.
- Local: timesfm/autots/jax/peft KHÔNG cài; wheel LightGBM local KHÔNG build CUDA → `gpu-probe` đầy đủ trên laptop sẽ dừng ở `BACKEND_GPU_FAILED` (đúng luật §10); dùng `--backends torch,xgboost` khi chỉ muốn kiểm cơ chế. torch 2.11+cu128, RTX 3050 Ti.
- Windows: `PYTHONPATH` tách bằng `;`; console cp1252 → `PYTHONUTF8=1` cho script ad-hoc.

## Pitfalls

- Data 2 năm: cột `ts` (không phải `timestamp`) — alias trong bộ nhớ, không sửa file; `datetime` ISO `+00:00`; LF phải dẫn xuất từ ĐÚNG file HF
  (sidecar sha) — HF khác → `check-data` hard-fail `LF_DERIVATION_MISMATCH`. pandas 3: epoch từ datetime tính bằng `total_seconds()` (resolution s/us/ns).
- B0: `timestamp_indices` chỉ check `t < end` → harness trừ 3'; Vast dùng LightGBM build CUDA; GPU histogram không bit-exact.
- TimesFM 2.0.2: `decode()` no_grad → `train_forward`; RMSNorm scale init 0; inject sau `load_checkpoint`; `torch_compile=False`; kênh 0 = mean head;
  k ≥ 1 covariate → phần dư OLS; jax `PREALLOCATE=false`.
- AutoTS 1.0.4: bug `sklearn.py:3337`; `max_windows` mặc định cắt FIT (FIT 172k bar → phải ≥ 200k như config); backend LightGBM theo config.
- C_short: `log_rv{k}_rv60`, `rsi1`, `bb_pctb_2`… NaN khi r1 = 0; tree nhận NaN, LSTM điền 0.
- `--smoke`/`--allow-cpu` chỉ với `dataset_label` `synthetic*`. Fold-parallel: run cần predictor sống chạy tuần tự.

## Important Files

- Repo GitHub (private): https://github.com/tson295/P0_forecasting — branch `main`; **data 2 năm (1m + 5m) ĐÃ push qua Git LFS**; experiments/** tracked (LFS nhị phân); CSV data khác vẫn không push.
- `configs/p0_full.json` (data 2 năm, `rolling_spread`, `experiments/full`, `prev_run_dir: experiments/15d`) · `configs/p0_15d.json` (lịch sử).
- `data/data_checksums_2y.json` (anchor), `data/BTC_5m_2y.derivation.json` (sidecar LF), `data/data_checksums.json` (15 ngày).
- `src/p0/`: `data.py` (`read_ohlcv_csv` alias ts, `derive_lf_5min`, `write_lf_csv`), `split.py` (`RollingSpec`, `make_rolling_spread`),
  `cli.py` (check-data / derive-lf / lock-s0 / loop / tfm-final / autots-search / champion-replay / orchestrate / gpu-probe / ensemble /
  final / visualize), `features_short.py`, `s0.py`, `checker_log.py`, **`gpu.py`** (chính sách thiết bị + bind CUDA_VISIBLE_DEVICES),
  **`scheduler.py`** (worker/queue/dispatch + `scheduler_log.jsonl`), `fold_parallel.py` (adapter), **`orchestrate.py`** (DAG nhánh + replay),
  `lora.py`, `models_tfm.py`, `visualize.py`.
- `scripts/`: `checker_record.py`, `canary_lora.py`, `vast_canary.py`, `canary_xreg_gpu.py`, `vast_bootstrap.sh`.
- `.claude/agents/`: `checker.md`, `run-monitor.md` (mới 2026-09-04d), `infra.md`, `analyst.md` (hậu-run), `researcher.md` (dormant).
- `docs/RESEARCH_PLAN.md` rev 10.4 · `docs/reference/audit_timesfm_lora.md`, `audit_timesfm.md`, `audit_autots.md`. `experiments/15d/` — vòng 15 ngày.

## Open Questions

- Thời gian/VRAM thật của LoRA (172,7k cửa sổ/fold) và XReg 163 candidate trên RTX 5000 Ada — canary sau unlock; có giới hạn `short_candidates` — user quyết.
- Với 2 GPU × 1 slot: hai task nặng cùng lúc có đủ VRAM cho nhánh tfm (LoRA batch 64) không — đo bằng `canary_lora.py`; nếu OOM thì chạy nhánh tfm với `P0_GPU_DEVICES=0` (1 task nặng), KHÔNG đổi batch nếu không bắt buộc.
- Có mở cross-asset (ETH/SOL/XRP) làm feature ở vòng sau hay không — user quyết.
