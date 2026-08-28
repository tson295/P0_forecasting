# MEMORY — trạng thái (update/replace, không append mâu thuẫn)

PHASE: PLAN rev 8 (15 ngày, lọc B0 → B0* theo R1–R4, mọi model từ B0* với 15fixed_m riêng → F*_m, GPU-only) / CHỜ USER REVIEW / CHƯA CODE
TRAINING: LOCKED

## Current Task

2026-08-27: user đơn giản hóa plan; rev 5 = chạy trên snapshot 15 ngày hiện có (Vast), lọc 306 feature B0 → B0* trước Bước 2 (PI + standalone + MI + kiểm chứng), mỗi model tự chọn feature set riêng từ B0*, metric trên giá, champion log, visualize theo origin; scale data để sau. Đã viết `docs/RESEARCH_PLAN.md` rev 5 và `.claude/CLAUDE.md`; bản cũ 2026-08-24 lưu `docs/archive/`. Chưa code, chưa training, chưa cài package.

## Exact Next Step

1. User review plan rev 5 — đã chốt: fold §1.2, số vòng cố định §1.3, chạy Vast. Còn xem: lọc B0 §1.4 (tier/kiểm chứng), thứ tự model §2.2, §2.4, metric trên giá §0, log/visualize §7.
2. Sau duyệt: code tối thiểu (`src/data.py`, `split.py`, `features_ext.py`, `metrics.py`, `run_lgbm.py` + `tests/`); smoke CPU vài trăm dòng.
3. User unlock training → Vast: LightGBM §1.3 trên B0-306 → §1.4 lọc → B0* → §1.3 lại trên B0* → vòng lặp → XGBoost → CatBoost → TimesFM (audit API trước) → ExtraTrees → AutoTS (audit trước) → LSTM; champion log sau mỗi model → ensemble → Final TEST 2 ngày → all_models.csv + figure.
4. Phục hồi data đầy đủ + scale data: chỉ khi user quyết (plan §5).

2026-08-28: đã tạo `reports/smoke_visualize.py` + `reports/smoke_visualize.md` + `reports/smoke/*.png` (layout mẫu, SỐ GIẢ, không phải kết quả); đã viết lại 8 agent files + `AGENT.md` theo plan rev 6 (remote-infra giữ nguyên).

## Decisions (mới nhất trước)

- 2026-08-28 (rev 8, user chốt): (a) ExtraTrees → XGB-RF. (b) Cờ +/− khi lọc B0: > 0 ở ≥ 2/3 horizon. (c) B0* (bộ tốt nhất từ B0-306 và R1–R4) là điểm xuất phát CHUNG cho mọi model; từ B0* mỗi model chạy ES một lần → 15fixed_m riêng (15fixed_306 chỉ cho R1–R4), rồi tự feature search bằng chính model đó → F*_LGBM, F*_XGB, F*_Cat, … có thể khác nhau; KHÔNG để LightGBM tìm F* trước rồi model khác kế thừa. Confirmation F*_m: ES bật, 3 seed (ghi best_iteration cho Final). ε_m đo trên B0* ngay sau calibrate (LightGBM thêm ε trên B0-306 cho lọc).
- 2026-08-28 (rev 7): training chỉ GPU, cấm CPU training → ExtraTrees (sklearn) thay bằng XGB-RF (XGBoost random-forest mode GPU); AutoTS regression_model LightGBM/XGBoost GPU; standalone 1-feature chạy GPU. Lọc B0 §1.4 bỏ tier: cờ PI+/SA+/MI+ = > 0 ở ≥ 1 horizon; R1 = PI+∨SA+∨MI+, R2 = PI+∨(SA+∧MI+), R3 = PI+, R4 = SA+; 4 run kiểm chứng vs B0-306, chọn bộ không tệ hơn có MedianGain cao nhất; mỗi cột có giữ/bỏ theo từng R trong b0_filter.csv. Chỉ MedianGain (với ε) quyết định mọi lựa chọn; WinRate/P10/Worst báo cáo. Ensemble = champion + mọi model có MedianGain vs E0 > 0 (equal và 1/MSE), so với champion bằng luật §3. Latency báo cáo p95/p99/max (bỏ p50), ghi train device (GPU) và predict device thực tế. Figure: palette categorical cố định đã validate (dataviz), mỗi model một màu + marker ở mọi figure.
- 2026-08-28 (rev 6): theo dõi inference latency (§7.4): predict một origin batch 1, per model × horizon, p50/p95/p99 (+mean/max); tree đo riêng từng h, model một lần gọi ra 3 bước gán chung (`shared`); pass riêng sau train ở confirmation F*_m và Final, warm-up 50, cuda synchronize, assert batch == batch-1; chưa gồm tính feature; chỉ theo dõi, không ảnh hưởng training/loss/KEEP-DROP/champion.
- 2026-08-27 (rev 5): lọc 306 feature B0 một lần trước Bước 2 (§1.4): PI (3 lần xáo trên VAL, dùng 15 model baseline), standalone 1-feature LightGBM gốc vs E0 và vs B0-306 (cờ đỏ nếu một cột thắng B0-306), MI regression trên z-target FIT với null xáo trộn; fail = fail ở cả 3 horizon; Tier 1 (fail cả ba) / Tier 2 (PI ≤ 0 + một tiêu chí) / R3 (chỉ giữ PI > 0); 3 run kiểm chứng vs B0-306 → B0* = bộ không tệ hơn có MedianGain cao nhất (hòa → nhỏ nhất); lọc không giúp → B0* = B0-306. File B0 không sửa; B0-306 vẫn log reference. Mọi model bắt đầu từ B0*; LSTM dùng fine feature còn trong B0*; AutoTS base regressor = B0*.
- 2026-08-27 (rev 4): fold §1.2 + số vòng cố định §1.3 = chốt. Giai đoạn 15 ngày chạy Vast (GPU detect, có thể 3090). Thứ tự model theo thời gian chạy tăng dần: LightGBM → XGBoost → CatBoost → TimesFM → ExtraTrees → AutoTS → LSTM (cuối). AutoTS tổng hợp F*_A1 ∪ F*_A2 chỉ sau khi cả hai vòng lặp xong (1 run/model). Metric tính trên giá (P̂ = C_t·exp(ŷ)); r/dir-acc trên thay đổi giá; predict vẫn log return. Champion: ban đầu LightGBM code gốc; sau mỗi model so sánh, log đổi/giữ (`champion_log.csv`); `keepdrop_<model>.csv` mỗi candidate; `all_models.csv` tổng hợp; visualize theo origin t → 3 điểm t+1..t+3 (Fig A), bar/heatmap (Fig B), theo ngày (Fig C). Candidate thua vì base 306 feature: Gain standalone diagnostic + KEEP khi không đổi + safety-net block cuối vòng lặp (§2.4).
- 2026-08-27 (rev 3): làm việc trên snapshot 15 ngày trước, scale data để sau. Fold: expanding FIT từ 01-19 02:46, ES = ngày trước VAL (trừ 60' purge), VAL 1 ngày × 5 (01-27→01-31), TEST 02-01→02-02 21:27. ES chỉ dùng ở run baseline mỗi model để lấy best_iteration; candidate dùng số vòng cố định (`fixed_rounds`). Thứ tự model: LightGBM → XGBoost → CatBoost → ExtraTrees → TimesFM → AutoTS → LSTM. TimesFM: TFM-POINT; vòng lặp covariate nếu API có; thắng E0 → LoRA. AutoTS chỉ hướng 1: 2 model cố định (WindowRegression/LightGBM, MultivariateRegression/ExtraTrees), mỗi model một vòng lặp, rồi thử tổng hợp F*_A1 ∪ F*_A2; bỏ hướng tự search. LSTM tự chạy vòng lặp (1 seed trong loop, 3 seed confirmation); dự phòng nếu hết thời gian: thử từng F*_m của model khác và chọn theo metric.
- 2026-08-27 (rev 2): **mỗi model chọn feature set riêng** (F*_m) bằng cùng vòng lặp add-one; chạy từng model một; không model nào dùng chung feature.
- 2026-08-27: 4-step plan. Feature thử lần lượt từng cột: tốt hơn/gần như không đổi → KEEP, tệ hơn (MedianGain < −ε_m, ε_m = nhiễu seed của model đó) → DROP. Bước 4: LightGBM, XGBoost, CatBoost, ExtraTrees (mới), TimesFM, AutoTS, LSTM; ensemble equal/1-MSE. Không thêm KNN/Bagging/LinearRegression/Lars; không yfinance. Near/Far → một TEST. Bỏ dossier/Wave-1/daily-block test/multiple-testing framework.
- 2026-08-27: `amount/volume` là VWAP thật theo trade trong bar (Binance quote volume; verify amount/volume ≈ price trong [L,H]) — sửa giả định "true VWAP không có".
- 2026-08-24: point-only objective; OHLCV-only; B0 frozen; metric Gain/MedianGain/WinRate/P10/Worst + RMSE/MAE/r/dir-acc.
- 2026-08-21: B0 frozen, Gain metric.

## Experiment Findings

(chưa có — chưa chạy experiment thật)

## Data / Implementation Blockers

- Snapshot 15 ngày đang dùng: `BTC_hf_1min.csv` 21.916 dòng (cắt đúng 2 MiB, dòng cuối cụt), 2026-01-18 16:15 → 02-02 21:30; B0-eligible 21.258 origin (01-19 02:46 → 02-02 21:27), warmup 631 bar; lưới/dup/gap/OHLC đã kiểm tra sạch. Regime: BTC −18% trong 15 ngày. `BTC_lf_5min.csv` đến 03-26 (đủ phủ). Data đầy đủ (manifest 289.320 bar đến 08-07) chưa phục hồi — để sau.
- lightgbm/xgboost/catboost/timesfm/autots chưa cài local (có torch 2.11+cu128, sklearn 1.8, scipy 1.17, pandas 3.0.3, numpy 2.4). Máy local có RTX 3050 Ti.
- Raw header lowercase vs B0 uppercase → adapter ở ingestion.
- LF 5-min nhãn T = gộp 1-min bars (T−4..T] (verify: LF 16:20 open = HF 16:16 open) → as-of join `T ≤ t` là causal.
- Repo chưa có commit nào (mọi file staged, nhiều file modified sau staging).

## Pitfalls

- B0: Huber alpha 0.9 trong z-space; ES monitor huber; `timestamp_indices` chỉ check `t < end` → harness phải trừ 3'; `fixed_rounds` có sẵn trong `fit_lgbm_baseline`; `require_p100=True` override bằng `LGBMConfig(require_p100=False)`; `device_type="gpu"` = build OpenCL (wheel pip mặc định không có); GPU histogram không đảm bảo bit-exact.
- pandas 3.0.3: CoW; dùng `'min'` thay `'T'`; index từ `pd.to_datetime(timestamp, unit='s', utc=True)`; `.to_numpy()`.
- Target h = 2, 3 chồng lấp → per-bar không iid (chỉ ghi nhớ khi đọc kết quả).
- Lag-1 autocorr 1-min ≈ −0.06 trên snapshot → tín hiệu điểm cỡ 0.1–0.2 pp RMSE ở h=1, ~0.03 pp ở h=3; Gain vài pp = nghi leakage. Forecast trông "phẳng" là bình thường.
- Directional accuracy: bỏ bar C_{t+h} = C_t (~3.7%).
- LightGBM/CatBoost predict luôn chạy CPU dù train GPU (đặc tính thư viện) → cột predict device trong latency là CPU; không phải CPU training.
- Metric trên giá: RMSE USD phụ thuộc mức giá (78k–95k trong 15 ngày); Gain là tỷ lệ nên ít bị ảnh hưởng; r/dir-acc phải tính trên thay đổi giá, không trên giá tuyệt đối.
- Chi phí trên 15 ngày: tree ≈ 1–2 h/model cho 39 candidate; AutoTS ≈ 2–4 h/model (lưới origin thưa mỗi 5'); LSTM ≈ 3–10 h (1 seed); tổng ≈ 12–25 h máy.

## Important Files

- Repo GitHub (private): https://github.com/tson295/P0_forecasting — branch `main`; raw CSV không push (.gitignore).
- `Baseline_LGBM.py` — B0 frozen, deny-protected.
- `docs/RESEARCH_PLAN.md` — plan rev 6 (canonical).
- `reports/smoke_visualize.md` + `reports/smoke_visualize.py` — layout mẫu bảng/figure với số giả (không phải kết quả).
- `.claude/CLAUDE.md` — invariants rút gọn.
- `docs/archive/` — plan/hiến pháp/memory cũ 2026-08-24 (tham khảo).
- `data/manifest.json` — range/row count dự kiến của data đầy đủ.
- `timesfm_ohlcv_distribution_forecasting_R0_R6.md`, `g_research_crypto_solutions_summary.md` — reference only.

## Open Questions

- TimesFM package/checkpoint/backend và có covariate API (xreg) hay không — audit trước Bước 2 #5.
- AutoTS version, tên 2 model cố định (WindowRegression/LightGBM, MultivariateRegression/ExtraTrees), cách truyền regressor, rolling predict — audit trước Bước 2 #6.
- Có mở cross-asset (ETH/SOL/XRP) làm feature ở vòng sau hay không — user quyết.
