# MEMORY — trạng thái (update/replace, không append mâu thuẫn)

PHASE: **XONG TOÀN BỘ PIPELINE 15 NGÀY** (Phase A → B → C → final) trên Vast RTX 3090, 2026-09-01 23:38 UTC
TRAINING: UNLOCKED (đã chạy xong; TEST đã chạm ĐÚNG MỘT LẦN ở bước `final`)

## Current Task

**Full run 15 ngày ĐÃ XONG** (2026-08-31 20:49 → 2026-09-01 23:38 UTC, ~27 h máy).
Champion cuối = **xgbrf**; **KHÔNG ensemble** (chỉ 1 thành viên đủ điều kiện). TEST chạm 1 lần ở `final`.
Kết quả chính: xem "Experiment Findings". Việc còn lại do user quyết (§5 data đầy đủ, hoặc phân tích sâu).

## Exact Next Step

Không còn bước bắt buộc nào. Lựa chọn tiếp theo do user quyết:
1. Gọi agent `analyst` đọc kết quả thật (`experiments/summary/all_models_test.csv`, `champion_log.csv`,
   `keepdrop_*.csv`, figure §7.3) → anomaly/regime/đề xuất experiment kế tiếp.
2. §5: phục hồi data đầy đủ (289.320 bar) → kiểm lại B0*/F*_m trên regime khác + scale data → TEST 30 ngày.
3. Tắt instance Vast để dừng tính tiền (kết quả đã commit; /workspace KHÔNG phải volume → phải kéo về trước).

## Decisions (mới nhất trước)

- 2026-09-01 (user chốt, sửa lại quyết định cũ): **invariant "training chỉ GPU" áp dụng cho CẢ bước xreg của TimesFM**, không chỉ mạng neural. Bỏ đề xuất (a) của `audit_timesfm.md` §9.1 (jax[cpu] + `force_on_cpu=True`); dùng (b): cài `jax[cuda12]==0.11.1` + `xreg_force_on_cpu=False`. Trên máy này KHÔNG có xung đột wheel: torch cu128 đã cung cấp sẵn mọi nvidia-* mà jax cần (cudnn 9.19 thoả `<10,>=9.8`), pip chỉ thêm jax-cuda12-pjrt/plugin + nvidia-cuda-nvcc. BẮT BUỘC `XLA_PYTHON_CLIENT_PREALLOCATE=false` (jax mặc định chiếm ~75% VRAM → bóp chết model torch của TimesFM). Giới hạn còn lại phải nói rõ: `create_covariate_matrix` trong timesfm là numpy/sklearn thuần CPU, không có tuỳ chọn GPU — phần đưa lên GPU là đúng bước ước lượng beta_hat. Bằng chứng: `scripts/canary_xreg_gpu.py` chặn `jnp.linalg.pinv` bên trong `xreg_lib` và đọc `.devices()` của beta_hat = {'gpu'} (không suy diễn từ env). Kết quả: 2,9× nhanh hơn; số học lệch 5,2e-06 tuyệt đối (corr 0,9999931) — khác biệt float32 giữa backend, KHÔNG bit-exact.
- 2026-08-31 (TimesFM + AutoTS, user chốt lại — bản cuối): **TimesFM có ĐÚNG HAI nhánh feature selection**, mỗi nhánh chạy đủ protocol §2.1 (add-one → prune PI → confirmation): `tfm_b0` xuất phát S = B0\*, `tfm_ext` xuất phát S = ∅ (baseline = native trên r1). `tfm-final` so TFM_B0_best vs TFM_EXT_best bằng metric project → TimesFM-final → champion/ensemble/Final. **Bỏ** 3-way covariate strategy, `b0star_subset`, và việc freeze `ext_only` như lựa chọn duy nhất (audit §12 vẫn giữ làm lịch sử nghiên cứu, không còn là quyết định). Giữ 2 bug fix kỹ thuật: cùng head mean `quantile[...,0]` cho cả native lẫn covariate; đường covariate compile `per_core_batch_size=1`; 1 origin/lời gọi; covariate dịch 1 bar.
- **AutoTS bỏ stage union**: WR/MR cố định là **probe** (mỗi cái từ B0\*, add-one → prune → confirmation → F_WR_best / F_MR_best). `autots-search` chạy framework AutoTS **riêng cho từng bộ** (dedup nếu trùng), template GPU do ta khai báo + `max_generations=0`, search chỉ trên training-side FIT+ES, freeze template rồi rolling predict outer VAL; so `result_WR` vs `result_MR` bằng metric project → AutoTS-final → champion/ensemble/Final. Probe không so champion, không vào ensemble, không refit ở Final.
- 2026-08-31 (agent, user chốt): còn **4 agent** — `checker` (verify độc lập + phủ quyết), `researcher` (audit API/version + verdict methodology), `analyst` (sau full run: anomaly/failure/regime + đề xuất experiment/feature), `infra` (GPU/env troubleshooting khi bootstrap fail). Bỏ `main-controller`, `coder`, `runner`: pipeline đã deterministic và CLI tự ép luật (TRAINING lock, GPU preflight, checksum §6.1, `--smoke` chỉ cho synthetic, `loop` đầu tiên phải là lgbm), bước hiện tại đọc ở plan §8 + MEMORY, mỗi bước là một lệnh `python run.py`, còn viết code cần full context nên do session chính làm.
- 2026-08-31 (visualize): thêm **Fig T — trajectory** (3 ảnh h=1,2,3) ở ĐÚNG hai chỗ đã có figure: sau khi win_m so với champion (VAL, mỗi fold một đoạn, không nối qua ranh giới fold) và Final (TEST, mọi model, 2 nhóm). Vẽ giá BTC thô: `actual_h(t) = C[t+h]`, `pred_h(t) = C[t]·exp(ŷ_h(t))`, trục x = timestamp t+h. Fig P / Fig HM / heatmap giữ nguyên, không gộp.
- 2026-08-31 (seed/ε, user chốt): **ba vai trò seed tách bạch** ở CẢ phase A và B. `calib_seed = 8586` CHỈ dùng cho run ES lấy số vòng/epoch cố định. `eval_seeds = 8587/8588/8589` chạy với số vòng cố định để đo ε và làm confirmation 3 seed. **ε mới**: mỗi ô (fold, horizon) có R1,R2,R3 → `mu = mean`, `sigma = std(ddof=0)`, `noise_cell = 100·sigma/mu` (pp); `ε = max(0.005, sqrt(mean(noise_cell²)))` (RMS 15 ô) — KHÔNG seed nào làm mốc/mẫu số (bỏ hẳn cách cũ "Gain của seed k vs seed 0"). **`selection_seed = 8587` (mặc định `eval_seeds[0]`) dùng cho MỌI bước selection**: PI/SA/MI + 4 run R1–R4 (phase A), baseline B0* + 39 candidate add-one + prune PI (phase B), refit Final — một giá trị duy nhất, không đổi giữa các Rk/candidate, để chênh lệch RMSE chỉ do feature set. Run baseline của §1.4 nay dùng `15fixed_306` tại selection_seed (thay vì ES) và chính là mốc RMSE cho R1–R4, cũng là 15 model dùng cho PI.
- 2026-08-29 (visualize + model mới): Fig P thay Fig H_h — forecast path của MỘT origin (x = t..t+3, y = P̂ − C_t), 3 origin đại diện vol thấp/trung bình/cao chọn bằng quy tắc cố định (VAL: origin ≥ 12:00 UTC; TEST: origin đầu khối 60'), không chọn theo error; file `fig_path_<model>_vs_champion.png`, `fig_final_paths_all_models.png`. Prune PI chạy trên mọi loại input: tabular (cột), LSTM (kênh của cửa sổ 512'), TimesFM/AutoTS (cột covariate/regressor) — cùng một định nghĩa PI. TimesFM/AutoTS trả thẳng log-return (`FitResult.is_logret=True`), không qua TargetTransform. AutoTS chấm trên **toàn bộ origin** (bỏ lưới thưa 5' để §3 so cùng 15 ô). XGB-RF dùng `XGBRegressor(n_estimators=1, num_parallel_tree=N)` thay `XGBRFRegressor` (deprecated từ xgboost 3.4) — bit-exact.
- 2026-08-29: B0 `TargetTransform` có bug nhân in-place `(n,1) *= (1,3)` (ValueError ở mọi numpy) → B0 gốc không chạy nguyên bản; file B0 không sửa; harness dùng `src/p0/transform.py` tái hiện đúng công thức (unit test so công thức + test ghi nhận bug). LightGBM trong harness = `_make_model`/`LGBMConfig`/ES huber y hệt `fit_lgbm_baseline`. Prune PI: bỏ cột ext không có cờ PI+ (PI > 0 ở ≥ 2/3 horizon). Latency §7.4: thread mặc định thư viện (ghi cột), assert |batch − batch-1| ≤ 1e-6. TEST = 2.728 origin (plan cũ ghi 2.725 — sai số cộng).
- 2026-08-28 (rev 9b, user chốt): gộp 3 seed khi confirmation = mỗi configuration (F*_m, F*_m^prune) 3 seed → 3 bảng RMSE 15 ô → mỗi ô lấy MEAN RMSE 3 seed → bảng RMSE̅; Gain_{f,h} = 1 − RMSE̅^prune/RMSE̅^unprune; MedianGain = median 15 Gain; ≥ −ε_m → prune, thấp hơn → unpruned. Cùng cách gộp cho win vs champion (bảng RMSE̅ hai bên). Cửa sổ visualize: 3 ngày VAL/fold khác nhau theo std r1 trong ngày = thấp / trung bình / cao, cửa sổ 12:00–13:00 UTC; TEST: 3 cửa sổ 60' theo std r1 thấp nhất / trung vị / cao nhất.
- 2026-08-28 (rev 9, user chốt): bỏ safety-net; sau vòng lặp chỉ prune PI (bỏ cột ext PI ≤ 0) → F*_m^prune; confirmation 3 seed (cách gộp median từng ô — đã thay bằng mean RMSE từng ô ở rev 9b); win_m = F*_m^prune nếu ≥ −ε_m so với F*_m, ngược lại F*_m; win_m so với champion cùng cách (median từng ô). Visualize sau mỗi model: win vs champion — mỗi horizon 3 ảnh (3 cửa sổ 60 origin) + 2 heatmap 15 ô; Final: heatmap TEST mọi model (khối 6h × h) + Fig H_h mọi model; actual đen; win = blue, champion = red; nhiều model → màu cố định, ≤ 8 màu/panel.
- 2026-08-28 (rev 8, user chốt): (a) ExtraTrees → XGB-RF. (b) Cờ +/− khi lọc B0: > 0 ở ≥ 2/3 horizon. (c) B0* (bộ tốt nhất từ B0-306 và R1–R4) là điểm xuất phát CHUNG cho mọi model; từ B0* mỗi model chạy ES một lần → 15fixed_m riêng (15fixed_306 chỉ cho R1–R4), rồi tự feature search bằng chính model đó → F*_LGBM, F*_XGB, F*_Cat, … có thể khác nhau; KHÔNG để LightGBM tìm F* trước rồi model khác kế thừa. Confirmation F*_m: ES bật, 3 seed (ghi best_iteration cho Final). ε_m đo trên B0* ngay sau calibrate (LightGBM thêm ε trên B0-306 cho lọc). LSTM cũng calibrate ES theo epoch trên B0* → fixed_epoch_LSTM cho mọi candidate (confirmation ES bật); XGB-RF (1 vòng cố định), TimesFM zero-shot, AutoTS xử lý theo cơ chế riêng, không ép fixed_rounds.
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

(dataset 15d `btc_1min_15d_2026-01-18_02-02`, Vast RTX 3090; VAL = 5 fold × 1 ngày, TEST = 02-01→02-02, 2.728 origin)

### KẾT LUẬN LỚN NHẤT: tín hiệu ~0 — gần như mọi model THUA E0
Trên VAL (MedianGain vs E0, 15 ô), **chỉ xgbrf > 0**: +0.0323 pp. Còn lại đều âm:
cat −0.0017 · xgb −0.0194 · lgbm −0.0270 · lstm −0.5291 · TimesFM-final −1.9958 · AutoTS-final −2.0578.
⇒ `ensemble` KHÔNG chạy được vì chỉ có 1 thành viên đạt (luật §3: cần ≥ 2). Đây là kết quả HỢP LỆ,
không phải lỗi pipeline. Nhất quán với mọi chẩn đoán dọc đường: lag-1 autocorr ≈ −0.06, ES dừng ở
1–63 vòng (tree) và 1–5 epoch (LSTM), B0-306 nằm đè lên E0.

### TEST (2.728 origin) — Gain vs E0 theo horizon (pp)
| model | h1 | h2 | h3 | dir-acc h1 | r h1 | latency p95 h1 |
|---|---|---|---|---|---|---|
| lgbm | **+0.247** | **+0.108** | +0.034 | 0.5191 | 0.0714 | 1.59 ms |
| lstm | +0.156 | +0.095 | **+0.457** | 0.5224 | 0.0685 | 0.89 ms |
| b0_306 | +0.233 | −0.067 | −0.023 | 0.5209 | 0.0686 | 1.62 ms |
| b0_star | +0.149 | +0.063 | −0.065 | 0.5081 | 0.0582 | 1.60 ms |
| cat | +0.111 | −0.119 | −0.116 | 0.5132 | 0.0516 | 1.51 ms |
| **xgbrf** (champion) | +0.088 | −0.040 | −0.142 | 0.5173 | 0.0451 | 0.58 ms |
| xgb | +0.086 | −0.010 | −1.075 | 0.5125 | 0.0517 | 0.46 ms |
| tfm | −1.367 | −1.840 | −2.914 | 0.4960 | −0.0217 | 324.46 ms |
| autots | −2.037 | −2.376 | −2.853 | 0.5224 | 0.0377 | 3.80 ms |
E0 RMSE TEST = 87,25 / 121,31 / 150,44 USD. dir-acc mọi model ≈ 0,49–0,52 (đồng xu). r ≈ 0,05–0,07.

### KHOẢNG CÁCH VAL → TEST (quan trọng khi đọc champion)
Champion chọn trên VAL là **xgbrf**, nhưng trên TEST **lgbm và lstm tốt hơn** ở cả 3 horizon.
Mọi chênh lệch ≤ 0,5 pp — nằm trong nhiễu, nên đây là minh hoạ giới hạn của việc chọn model khi
tín hiệu ~0, KHÔNG phải bằng chứng lgbm "thật sự" tốt hơn. Không được sửa gì sau khi xem TEST (§4).

### ε_m CHI PHỐI KEEP/DROP hơn cả chất lượng feature
ε: xgbrf 0,0200 · autots_mr 0,0050 (sàn) · lgbm 0,0966 · cat 0,0914 · xgb 0,2822 · lstm 0,4041 ·
autots_wr 0,6807 · TimesFM 0,0050 (sàn, tất định) · AutoTS-final 1,1663.
KEEP/DROP theo model: lgbm/xgb/cat/lstm/autots_wr **39 KEEP / 0 DROP**; xgbrf 31/8; autots_mr **8 KEEP / 31 DROP**;
tfm_b0 và tfm_ext **0 KEEP / 39 DROP**. Cùng data, cùng feature — khác nhau CHỈ vì sàn nhiễu từng model.
Bước lọc thật sự là **prune PI**: lgbm 14/40 · xgb 11/40 · cat 5/40 · xgbrf 12/32 · lstm 23/40 ·
autots_wr 21/40 · autots_mr 5/8 (nhưng confirmation chọn UNPRUNE).

### TimesFM: covariate làm HỎNG, native đỡ hơn nhiều
tfm_b0 (72 covariate B0\*) −17,73 pp vs E0; tfm_ext (native, 0 covariate) −1,9958 pp → **TimesFM-final = native**.
Nguyên nhân đã kiểm chứng: xreg in-context fit 72 regressor yếu trên cửa sổ 512 điểm → β nhiễu, sai số
dôi ra tăng TUYẾN TÍNH theo h (2,3e-4 → 4,4e-4 → 6,7e-4) trong khi E0 chỉ tăng ~√h. Thiết kế hai nhánh §2.2 #4
đã phân xử đúng. AutoTS-final = F_WR_best|wr:60 (21 cột ext), thắng F_MR_best (−2,06 vs −3,18 pp).

### Không có dấu hiệu leakage
Không ô nào đạt ngưỡng nghi ngờ > 1 pp so với E0/B0 theo hướng dương. Gain dương lớn nhất trên TEST là
+0,457 pp (lstm h3). Mọi Gain âm lớn đều đã truy được nguyên nhân (xreg của TimesFM/AutoTS).

## Data / Implementation Blockers

- Snapshot 15 ngày đang dùng: `BTC_hf_1min.csv` 21.916 dòng (cắt đúng 2 MiB, dòng cuối cụt), 2026-01-18 16:15 → 02-02 21:30; B0-eligible 21.258 origin (01-19 02:46 → 02-02 21:27), warmup 631 bar; lưới/dup/gap/OHLC đã kiểm tra sạch; sha256 ở `data/data_checksums.json` (CLI verify mọi bước). Số origin thực tế (`check-data`): FIT 9.887 / 11.327 / 12.767 / 14.207 / 15.647 (Final 17.087), ES 1.377, VAL 1.437, TEST 2.728 — B0 loại 24 origin ngày 01-24 08:30–09:33, 19:03–20:06 và 01-25 07:27–08:30 (3 bar bất thường lan theo lag 1/2/4/8/16/32/63; chỉ trong FIT, đặc tính B0, không sửa). Regime: BTC −18% trong 15 ngày. `BTC_lf_5min.csv` đến 03-26 (đủ phủ). Data đầy đủ (manifest 289.320 bar đến 08-07) chưa phục hồi — để sau.
- Local đã cài lightgbm 4.7.0 (CPU wheel), xgboost 3.4.1, catboost 1.2.10, pytest 9.1 — chỉ cho unit test; timesfm/autots chưa cài (chờ audit). torch 2.11+cu128, sklearn 1.8, scipy 1.17, pandas 3.0.3, numpy 2.4. Máy local có RTX 3050 Ti (training vẫn trên Vast).
- Raw header lowercase vs B0 uppercase → adapter ở ingestion.
- LF 5-min nhãn T = gộp 1-min bars (T−4..T] (verify: LF 16:20 open = HF 16:16 open) → as-of join `T ≤ t` là causal.
- Windows: `PYTHONPATH` tách bằng `;` (không `:`); console cp1252 không in được `→` → `run.py` tự reconfigure stdout UTF-8; script ad-hoc cần `PYTHONUTF8=1`.

## Pitfalls

- B0: Huber alpha 0.9 trong z-space; ES monitor huber; `timestamp_indices` chỉ check `t < end` → harness phải trừ 3'; `fixed_rounds` có sẵn trong `fit_lgbm_baseline`; `require_p100=True` override bằng `LGBMConfig(require_p100=False)`; `device_type="gpu"` = build OpenCL (wheel pip mặc định không có); GPU histogram không đảm bảo bit-exact.
- pandas 3.0.3: CoW; dùng `'min'` thay `'T'`; index từ `pd.to_datetime(timestamp, unit='s', utc=True)`; `.to_numpy()`.
- Target h = 2, 3 chồng lấp → per-bar không iid (chỉ ghi nhớ khi đọc kết quả).
- Lag-1 autocorr 1-min ≈ −0.06 trên snapshot → tín hiệu điểm cỡ 0.1–0.2 pp RMSE ở h=1, ~0.03 pp ở h=3; Gain vài pp = nghi leakage. Forecast trông "phẳng" là bình thường.
- Directional accuracy: bỏ bar C_{t+h} = C_t (~3.7%).
- TimesFM 2.5: `point_forecast` là q50, RMSE cần **mean** → dùng `quantile[..., 0]`; covariate BẮT BUỘC 1 origin/lời gọi (xreg fit chung `beta_hat`) và dịch 1 bar; timesfm 1.x không cài được trên Python 3.12.
- AutoTS 1.0.4 có bug `sklearn.py:3337` (`future_regressor.reindex(df)`) → không gọi được `fit_data(df, future_regressor=...)`; ta gọi `fit_data(df)` rồi tự gán `regressor_train` (chỉ cho MR; WR predict chỉ dùng `future_regressor.tail(1)`). `max_windows` mặc định 5000 cắt mất phần lớn FIT; nhánh xgboost không tự set seed.
- lightgbm ≥ 4.7 cảnh báo `eval_set` deprecated (B0 dùng eval_set) — harness lọc warning, giữ API cho đồng nhất với B0; nếu Vast cài lightgbm mới hơn bỏ hẳn eval_set thì sửa `models.LGBMModel` (không sửa B0).
- `--smoke`/`--allow-cpu` chỉ có tác dụng với `dataset_label` `synthetic*`; trên data thật CLI thoát — không tìm cách lách (cấm training CPU).
- LightGBM/CatBoost predict luôn chạy CPU dù train GPU (đặc tính thư viện) → cột predict device trong latency là CPU; không phải CPU training.
- Metric trên giá: RMSE USD phụ thuộc mức giá (78k–95k trong 15 ngày); Gain là tỷ lệ nên ít bị ảnh hưởng; r/dir-acc phải tính trên thay đổi giá, không trên giá tuyệt đối.
- Chi phí trên 15 ngày: tree ≈ 1–2 h/model cho 39 candidate; AutoTS ≈ 2–4 h/model (lưới origin thưa mỗi 5'); LSTM ≈ 3–10 h (1 seed); tổng ≈ 12–25 h máy.

## Important Files

- Repo GitHub (private): https://github.com/tson295/P0_forecasting — branch `main`; raw CSV không push (.gitignore).
- `docs/reference/audit_timesfm.md`, `docs/reference/audit_autots.md` — audit API/version (2026-08-29), căn cứ của `models_tfm.py` / `models_autots.py`.
- `src/p0/` + `run.py` — harness (§8); `configs/p0_15d.json`; `tests/`; `docs/VAST_SESSION_PROMPT.md`; `scripts/vast_bootstrap.sh`; `requirements.txt`; `data/data_checksums.json` (anchor §6.1).
- `Baseline_LGBM.py` — B0 frozen, deny-protected (có bug TargetTransform, xem Decisions 2026-08-29).
- `docs/RESEARCH_PLAN.md` — plan rev 9b (canonical; §8 = trạng thái code + lệnh chạy).
- `reports/smoke_visualize.md` + `reports/smoke_visualize.py` — layout mẫu bảng/figure với số giả (không phải kết quả).
- `.claude/CLAUDE.md` — invariants rút gọn.
- `docs/archive/` — plan/hiến pháp/memory cũ 2026-08-24 (tham khảo).
- `data/manifest.json` — range/row count dự kiến của data đầy đủ.
- `docs/reference/` — TimesFM R0–R6 (cũ), G-Research summary; audit API ghi vào đây. Reference only.
- `.claude/agents/` — 4 agent: checker, researcher, analyst, infra (registry `.claude/AGENT.md`; bỏ main-controller/coder/runner 2026-08-31).

## Open Questions

- TimesFM package/checkpoint/backend và có covariate API (xreg) hay không — audit trước Bước 2 #5.
- AutoTS version, tên 2 model cố định (WindowRegression/LightGBM, MultivariateRegression/ExtraTrees), cách truyền regressor, rolling predict — audit trước Bước 2 #6.
- Có mở cross-asset (ETH/SOL/XRP) làm feature ở vòng sau hay không — user quyết.
