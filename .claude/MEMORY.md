# MEMORY — trạng thái (update/replace, không append mâu thuẫn)

PHASE: **DATA 2 NĂM ĐÃ WIRE + SPLIT ROLLING_SPREAD + S0/Candidate_m TRÊN DATA THẬT — chờ preflight Vast + user unlock**, 2026-09-04
TRAINING: LOCKED

## Current Task

Vòng **expanded-data** trên **data 2 năm thật** (quyết định user 2026-09-04b): `data/BTC_1m_2y.csv` (1.051.201 bar 60 s, 2024-09-03 16:29 →
2026-09-03 16:29 UTC, sha256 `559ce040…f097`) + LF 5' dẫn xuất tất định `data/BTC_5m_2y.csv` (210.239 bar, sha256 `0e5fb9ad…f2fef`).
Split `rolling_spread`: 5 VAL 3 ngày rải đều 2025-01-07 → 2026-08-01, FIT 120 ngày rolling + ES 5 ngày, TEST 30 ngày cuối (08-04 16:30 → 09-03 16:30).
`lock-s0` đã chạy trên data thật: Candidate_m = 163 cho mọi model (0 overlap với S0_m). **Chưa chạy training nào, chưa chạm TEST.**
Methodology không đổi so với commit hiệu chỉnh `afca1c8` (S0 khoá toàn bộ, C_short 163, TimesFM-LoRA native → XReg, AutoTS WR/MR, checker không tương tác…).

## Exact Next Step

1. Trên Vast: clone (git lfs install), scp `data/BTC_1m_2y.csv` (hoặc scp cả `data/BTC_5m_2y.csv` + `.derivation.json` — nếu không thì
   `python run.py derive-lf --config configs/p0_full.json` dẫn xuất lại, phải ra đúng sha `0e5fb9ad…`), rồi:
   `bash scripts/vast_bootstrap.sh` → `scripts/vast_canary.py` + `scripts/canary_xreg_gpu.py` →
   `python run.py check-data --config configs/p0_full.json` (verify anchor `data/data_checksums_2y.json`, in 5 fold + final như dưới) →
   `python run.py lock-s0 --config configs/p0_full.json` (phải ra 163/model, 0 overlap; audit label = `btc_1min_2y_2024-09-03_2026-09-03`) →
   `pytest -q -x` → `python scripts/checker_record.py --exp experiments/full --blocking` sạch ERROR → agent `checker` (không tương tác).
2. **User unlock rõ ràng** → `scripts/canary_lora.py` (1 fold × 1 epoch × 64 origin, đo thời gian/VRAM; FIT 172.7k cửa sổ/fold) →
   `docs/VAST_SESSION_PROMPT.md` (loop lgbm → xgb → cat → tfm → tfm-final → xgbrf → autots_wr → autots_mr → autots-search → lstm → ensemble → final → visualize).

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
- `experiments/full/`: `s0/` (data thật), `checker_log.jsonl`; chưa có wins/lora/runs/final.
- Snapshot 15 ngày vẫn ở `data/BTC_hf_1min.csv` + `data/BTC_lf_5min.csv` (config `configs/p0_15d.json`, anchor `data/data_checksums.json`) — lịch sử.
- Local: timesfm/autots/jax/peft KHÔNG cài; test TimesFM-LoRA dùng stub + canary local trên sdist random-init. torch 2.11+cu128, RTX 3050 Ti.
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

- Repo GitHub (private): https://github.com/tson295/P0_forecasting — branch `main`; raw CSV không push; experiments/** tracked (LFS nhị phân).
- `configs/p0_full.json` (data 2 năm, `rolling_spread`, `experiments/full`, `prev_run_dir: experiments/15d`) · `configs/p0_15d.json` (lịch sử).
- `data/data_checksums_2y.json` (anchor), `data/BTC_5m_2y.derivation.json` (sidecar LF), `data/data_checksums.json` (15 ngày).
- `src/p0/`: `data.py` (`read_ohlcv_csv` alias ts, `derive_lf_5min`, `write_lf_csv`), `split.py` (`RollingSpec`, `make_rolling_spread`),
  `cli.py` (check-data / derive-lf / lock-s0 / loop / tfm-final / autots-search / ensemble / final / visualize), `features_short.py`, `s0.py`,
  `checker_log.py`, `fold_parallel.py`, `lora.py`, `models_tfm.py`, `visualize.py`.
- `scripts/`: `checker_record.py`, `canary_lora.py`, `vast_canary.py`, `canary_xreg_gpu.py`, `vast_bootstrap.sh`.
- `docs/RESEARCH_PLAN.md` rev 10.2 · `docs/reference/audit_timesfm_lora.md`, `audit_timesfm.md`, `audit_autots.md`. `experiments/15d/` — vòng 15 ngày.

## Open Questions

- Thời gian/VRAM thật của LoRA (172,7k cửa sổ/fold) và XReg 163 candidate trên RTX 3090 — canary sau unlock; có giới hạn `short_candidates` — user quyết.
- Có mở cross-asset (ETH/SOL/XRP) làm feature ở vòng sau hay không — user quyết.
