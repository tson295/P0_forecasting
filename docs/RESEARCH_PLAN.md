# RESEARCH PLAN — BTC 1-phút point forecasting (bản đơn giản hóa)

Cập nhật: 2026-08-29 (code §8 + checker review: TEST = 2.728 origin, prune PI theo cờ PI+ ≥ 2/3 horizon, latency §7.4 thread mặc định thư viện, checksum §6.1 bắt buộc trong CLI) · 2026-08-28 (rev 9b: gộp 3 seed bằng mean RMSE từng ô; cửa sổ visualize ở 3 ngày VAL vol thấp/trung bình/cao; rev 9: bỏ safety-net, chỉ prune PI; confirmation 3 seed → win_m; figure win vs champion + Final mọi model; rev 8: B0* là điểm xuất phát chung của mọi model; mỗi model calibrate 15fixed_m riêng trên B0* rồi tự feature search → F*_m; cờ + = > 0 ở ≥ 2/3 horizon; rev 7: training chỉ GPU, ExtraTrees → XGB-RF, lọc B0 theo R1–R4 không tier, ensemble theo skill vs E0, latency p95/p99/max, màu cố định; rev 6: latency §7.4; rev 5: lọc B0). Thay thế hoàn toàn roadmap 2026-08-24 (lưu tại `docs/archive/RESEARCH_PLAN_2026-08-24_detailed.md`, không còn hiệu lực).

Luồng nghiên cứu:

```
Fix dataset 15 ngày (5 fold VAL 1 ngày, TEST 2 ngày cuối)
→ Lọc 306 feature B0 bằng PI + standalone + MI → B0*  (§1.4)
→ Feature search: mỗi model (nhanh → chậm) từ CÙNG B0*: m(B0*) + ES → 15fixed_m → add-one 39 candidate → F*_m riêng
     LightGBM → XGBoost → CatBoost → TimesFM → XGB-RF → AutoTS (2 model cố định) → LSTM
→ Sau mỗi model: prune PI → 3 seed (mean RMSE từng ô → Gain 15 ô → MedianGain) → win_m → so với champion, log đổi/giữ, vẽ win vs champion
→ Ensemble → Final evaluation (TEST 2 ngày) → bảng tổng hợp + visualize
→ (để sau) data đầy đủ → scale data → TEST 30 ngày
```

Trạng thái: **code xong + unit test/smoke PASS + checker review đã sửa (2026-08-29); chưa training.** Training chỉ chạy trên Vast khi user nói unlock (§8).

---

## 0. Giữ nguyên từ thiết kế đã có (và các điểm sửa theo user)

- **Target**: `y_h(t) = log(C[t+h] / C[t])`, h = 1, 2, 3 phút. Model dự báo one-step return `r` phải cộng dồn `ŷ_h = Σ_{i≤h} r̂_{t+i}` trước khi chấm.
- **Data**: BTC 1-phút OHLCV + amount (Binance) `data/BTC_hf_1min.csv`; `data/BTC_lf_5min.csv` chỉ dùng cho feature ở resolution 5 phút. Không dùng `*_close.csv`, không cross-asset (thêm sau nếu user muốn).
- **Baseline** (các nghĩa, dùng xuyên suốt):
  - **B0-306** = 306 feature của `Baseline_LGBM.py` (file không sửa) chạy bằng LightGBM code gốc. Luôn log làm reference. Lưu ý: `TargetTransform` trong file B0 có bug nhân in-place (`(n,1) *= (1,3)` → ValueError) — harness dùng `src/p0/transform.py` tái hiện đúng công thức; mọi thứ khác (feature 306, `_make_model`, `LGBMConfig`, `build_lgbm_matrix`) import thẳng từ B0.
  - **Feature baseline = B0\*** = bộ tốt nhất chọn từ B0-306 và R1–R4 (§1.4, chọn cột trong harness). **Mọi model bắt đầu vòng lặp feature từ cùng B0\***; không model nào kế thừa feature set của model khác (§2.1).
  - **Model baseline / champion ban đầu** = LightGBM đúng code gốc (`fit_lgbm_baseline`, `LGBMConfig` không đổi: Huber alpha 0.9, TargetTransform `y / (rv60·√h)` fit train-only, seed 8586) trên F\*_LGBM. Gain của các model khác đo so với champion hiện tại (§3).
  - **E0** (ŷ = 0 ⇔ P̂ = C_t) luôn log.
- **Prediction và metric**: model dự báo **log return** `ŷ_h`; metric tính trên **giá**: `P̂_{t+h} = C_t · exp(ŷ_h)`, lỗi `e_h = P̂_{t+h} − C_{t+h}` (USD).
  - `RMSE_h`, `MAE_h` trên `e_h`; `Gain = 1 − RMSE_cand / RMSE_base` per horizon × fold; tóm tắt **MedianGain, WinRate, P10Gain, WorstGain** trên 15 ô (5 fold × 3 h). WinRate = tỷ lệ ô có Gain > 0; P10Gain = phân vị 10% của 15 ô (đuôi xấu, ~ô tệ thứ 2); WorstGain = ô tệ nhất.
  - **Chỉ MedianGain (so với ε) là tiêu chí quyết định** ở mọi chỗ: KEEP/DROP §2.1, chọn B0\* §1.4, đổi champion §3, thành viên ensemble §3. WinRate/P10Gain/WorstGain chỉ báo cáo để nhìn ổn định. PI/MI/standalone chỉ dùng để lập các bộ R1–R4 ở §1.4 (và permutation importance ở bước prune tùy chọn cuối vòng lặp), không tham gia KEEP/DROP của candidate.
  - Pearson r và directional accuracy tính trên **thay đổi giá** `P̂_{t+h} − C_t` so với `C_{t+h} − C_t` (r trên giá tuyệt đối vô nghĩa vì bị mức giá chi phối); directional accuracy bỏ bar có `C_{t+h} = C_t`.
  - Importance/permutation importance/MI chỉ là diagnostic hoặc bộ lọc §1.4, không phải metric quyết định. Loss/transform bên trong model không đổi (z-space Huber); chỉ evaluation trên giá.
- **Inference latency** (§7.4): thời gian predict **một origin** (batch 1) per model × horizon, tóm tắt p95/p99/max. **Chỉ để theo dõi** — không ảnh hưởng training, loss, KEEP/DROP, champion hay bất kỳ quyết định nào.
- **Split**: walk-forward FIT → ES → purge 60' → VAL; TEST cuối không chạm cho tới Final. 15 ngày: §1.2; data đầy đủ (để sau): §5.
- **Quy tắc biên**: origin t thuộc `[T_start, T_end)` chỉ khi `t ≥ T_start` và `t + 3' < T_end`. Feature chỉ dùng dữ liệu τ ≤ t. TargetTransform/scaler fit train-only mỗi fold.
- **Runtime**: giai đoạn 15 ngày chạy trên **Vast** (GPU detect tại runtime — có thể là RTX 3090, không hard-code; `LGBMConfig(require_p100=False)`). **Training chỉ trên GPU — cấm training bằng CPU, không CPU fallback**: LightGBM (build GPU), XGBoost (`device=cuda`), CatBoost (`task_type=GPU`), XGB-RF (XGBoost GPU), TimesFM/LSTM (torch GPU), model 1-feature standalone (LightGBM GPU), AutoTS với `regression_model` LightGBM/XGBoost cấu hình GPU (chốt khi audit). ExtraTrees sklearn không có GPU → thay bằng XGB-RF. CPU chỉ cho việc không phải training: tính feature, metric, MI/PI, unit/smoke test local, và predict của thư viện mặc định chạy CPU (LightGBM/CatBoost predict).

---

## 1. Bước 1 — Fix dataset (15 ngày hiện có) và lọc B0

### 1.1 Snapshot và kiểm tra (một lần)

`data/BTC_hf_1min.csv` hiện có 21.916 dòng, `2026-01-18 16:15 → 2026-02-02 21:30 UTC` (file bị cắt đúng 2 MiB, dòng cuối cụt → bỏ). Đã kiểm tra read-only: lưới 60 s, không duplicate, không gap, `H ≥ max(O,C)`, `L ≤ min(O,C)`, `amount/volume` nằm trong `[L, H]`. Trước khi chạy:

1. Adapter header lowercase → `Open/High/Low/Close/Volume` (không sửa B0); giữ `amount`.
2. Ghi checksum + số dòng + range vào `data/data_checksums.json`, nhãn `btc_1min_15d_2026-01-18_02-02`. Mọi kết quả giai đoạn này gắn nhãn dataset đó.
3. `BTC_lf_5min.csv` (đến 2026-03-26, đủ phủ 15 ngày): nhãn `T` = gộp 5 bar 1-phút `(T−4 … T]` → chỉ join bar có `T ≤ t`.
4. Lưu ý regime: 15 ngày này BTC 95.156 → 78.299 (−18%), vol cao; kết quả chọn feature/model là trên một regime, sẽ kiểm tra lại khi có data đầy đủ (§5).

B0-eligible origins: 21.258, `01-19 02:46 → 02-02 21:27` (warmup 631 bar).

### 1.2 Fold cho 15 ngày [đã chốt]

Expanding train, VAL = 1 ngày UTC, ES = ngày liền trước VAL (trừ 60' purge), TEST = 2 ngày cuối:

| Fold | FIT (expanding, từ 01-19 02:46) | ES | purge | VAL |
|---|---|---|---|---|
| 1 | → 01-25 23:56 (~9.9k origin) | 01-26 00:00 → 22:56 | 60' | 01-27 00:00 → 23:56 |
| 2 | → 01-26 23:56 (~11.3k) | 01-27 00:00 → 22:56 | 60' | 01-28 |
| 3 | → 01-27 23:56 (~12.8k) | 01-28 00:00 → 22:56 | 60' | 01-29 |
| 4 | → 01-28 23:56 (~14.2k) | 01-29 00:00 → 22:56 | 60' | 01-30 |
| 5 | → 01-29 23:56 (~15.7k) | 01-30 00:00 → 22:56 | 60' | 01-31 |
| TEST (refit) | → 01-30 23:56 (~17.1k) | 01-31 00:00 → 22:56 | 60' | 02-01 00:00 → 02-02 21:27 |

- Mỗi VAL = 1.437 origin; 5 fold = 7.185 origin, 15 ô. ES = 1.377 origin. TEST = 2.728 origin (02-01 Chủ nhật, 02-02 Thứ hai; origin cuối 02-02 21:27). FIT thực tế theo eligible của B0 (đã chạy `check-data` trên snapshot): 9.887 / 11.327 / 12.767 / 14.207 / 15.647 origin (fold 1–5), Final 17.087 — B0 loại 24 origin ngày 01-24/01-25 (3 bar bất thường lan theo lag), chỉ nằm trong FIT.
- Expanding thay vì rolling vì data ít; so sánh candidate vs baseline luôn trong cùng fold nên train size không làm lệch Gain.
- Mọi partition half-open, `t + 3' < T_end`.
- Cố định trong toàn bộ §1.4 và Bước 2, cho mọi model: cùng fold, cùng tập origin (= eligible của B0), cùng seed 8586, cùng config từng model.
- Lookback candidate ≤ 1440 phút → ext có thể NaN ở ngày đầu FIT; tree nhận NaN native; model không nhận NaN (LSTM, AutoTS tùy API) điền 0 sau chuẩn hóa train-only.

### 1.3 Nhiễu seed và số vòng cố định [đã chốt] — calibrate riêng cho từng phase và từng model

Nguyên tắc: **"số vòng cố định" = chính `best_iteration` mà early stopping dừng ở một run calibrate** (seed 8586, ES trên ES set, per fold × horizon → 15 giá trị) — không phải ước lượng thống kê. ES trên 1.377 dòng nhiễu, nên ES chỉ chạy **một lần cho mỗi (phase, model), trên đúng feature set của phase đó**; mọi run còn lại của phase dùng đúng 15 số vòng ấy (`fixed_rounds`, B0 hỗ trợ sẵn) để chênh lệch Gain chỉ do feature. **Không dùng chéo**: `15fixed_306` chỉ cho lọc B0; số vòng của LightGBM không dùng cho XGBoost/CatBoost và ngược lại.

Lịch calibrate:

| Phase | Feature set calibrate | Model | Kết quả | Dùng cho |
|---|---|---|---|---|
| A. Lọc B0 (§1.4) | B0-306 | LightGBM | `15fixed_306` (+ 15 model baseline dùng cho PI) + ε_LGBM(B0-306) | 4 run kiểm chứng R1–R4 → B0\* |
| B. Feature search (§2.1) | **B0\*** (chung cho mọi model) | từng model có early stopping: LightGBM, XGBoost, CatBoost (số vòng), LSTM (số epoch) — mỗi model một run | `15fixed_LGBM`, `15fixed_XGB`, `15fixed_Cat`, `fixed_epoch_LSTM` + ε_m | toàn bộ 39 candidate và prune PI của chính model đó (tính một lần, dùng chung cả phase) |
| C. Prune PI + confirmation (§2.1) | F\*_m và F\*_m^prune của chính model | model m, ES bật, 3 seed mỗi configuration | bảng `RMSE̅` 15 ô (mean 3 seed từng ô) mỗi configuration → Gain prune vs unprune từng ô → MedianGain → **win_m** (+ `best_iteration`/best epoch ghi lại cho Final refit) | so với champion (§3), figure §7.3 |

```
B0-306 + ES (LGBM) → 15fixed_306 → R1–R4 → B0*
   → LGBM(B0*) + ES → 15fixed_LGBM → LightGBM add-one 39 candidate → F*_LGBM
   → XGB(B0*)  + ES → 15fixed_XGB  → XGBoost  add-one 39 candidate → F*_XGB
   → Cat(B0*)  + ES → 15fixed_Cat  → CatBoost add-one 39 candidate → F*_Cat
   → LSTM(B0*) + ES theo epoch → fixed_epoch_LSTM → LSTM add-one 39 candidate → F*_LSTM
   (XGB-RF: 1 vòng boosting cố định; TimesFM: zero-shot; AutoTS: cơ chế riêng — cũng từ B0*, chỉ đo ε_m, không ép fixed_rounds)
```

- **LSTM** có epoch nên cũng calibrate: một run ES theo epoch trên B0\* (patience 5, ≤ 50 epoch) → `fixed_epoch_LSTM` per fold (head 3 output nên một số epoch cho cả 3 h); mọi candidate của LSTM train đúng số epoch ấy; confirmation bật lại ES.
- **XGB-RF** (1 vòng boosting cố định, `num_parallel_tree` cố định) không có gì để calibrate; **TimesFM** zero-shot không train; **AutoTS** cố định số vòng của regression_model bên trong theo cơ chế của AutoTS trong config. Ba model này xử lý theo cơ chế riêng, không ép khái niệm fixed_rounds; chỉ đo ε_m.
- **ε_m** đo ngay sau calibrate phase B của model đó: chạy `m` trên B0\* với 3 seed (8586, 8587, 8588), số vòng/epoch cố định vừa có (XGB-RF, AutoTS, TimesFM dùng config cố định của nó), 5 fold; Gain (trên giá) của seed 8587, 8588 so với 8586 trên 15 ô → 30 giá trị; `ε_m = max(0.005 pp, std của 30 giá trị)`. LightGBM đo thêm ε ở phase A trên B0-306 (dùng để chọn B0\*).

### 1.4 Lọc 306 feature của B0 → B0\* (một lần, trước Bước 2) [mới theo quyết định user]

Lý do: 306 cột của B0 (22 fine × 8 lag + 16 coarse × 8 lag + rv60/log_rv60) có thể chứa nhiều cột nhiễu; lọc một lần bằng LightGBM gốc trên 5 fold §1.2 rồi mới chạy Bước 2. `Baseline_LGBM.py` không đổi — lọc bằng chọn cột trong harness. B0-306 nguyên bản vẫn log làm reference ở mọi bảng.

Flow lọc:

```
B0-306 + ES (LightGBM, seed 8586) → 15fixed_306 + 15 model baseline
   ├─ (a) PI : xáo từng cột trong VAL × 3 lần → ΔRMSE giá per horizon (median 5 fold)
   ├─ (b) SA : LightGBM chỉ 1 cột, 5 fold × 3 h → Gain vs E0 và vs B0-306
   └─ (c) MI : mutual_info_regression(X_j, z_h) trên FIT − MI với target xáo trộn
→ cờ per cột: PI+ / SA+ / MI+  khi điểm số > 0 ở ≥ 2/3 horizon
→ R1 = PI+ ∨ SA+ ∨ MI+   R2 = PI+ ∨ (SA+ ∧ MI+)   R3 = PI+   R4 = SA+
→ 4 run kiểm chứng (LightGBM, 15fixed_306) so với B0-306 → MedianGain 15 ô
→ B0* = bộ không tệ hơn (≥ −ε_LGBM) có MedianGain cao nhất (hòa → nhỏ hơn); không bộ nào đạt → B0-306
→ experiments/b0_filter.csv: 306 dòng (điểm số, cờ, giữ/bỏ theo R1–R4) + 4 kết quả kiểm chứng
```

Ba điểm số cho từng cột `j` (per horizon; gộp 5 fold bằng median), tất cả trên giá:

**(a) Permutation importance (PI)** — dùng đúng 15 model B0-306 của run baseline §1.3 (seed 8586). Trên VAL mỗi fold: xáo trộn cột `j` giữa các origin VAL, predict lại, `PI_{j,h,f} = RMSE_perm − RMSE_gốc` (USD); lặp 3 lần xáo (seed khác nhau) lấy trung bình; median qua 5 fold. `PI ≤ 0` = xáo không làm model xấu đi → cột không được dùng hữu ích. Chi phí: chỉ predict, vài phút. Ghi thêm PI theo nhóm (xáo cùng lúc 8 lag của một base feature, 38 nhóm) để đọc, không dùng để quyết định.

**(b) Standalone 1-feature** — với từng cột `j`: LightGBM code gốc (cùng config, cùng TargetTransform, ES trên ES set) chỉ trên `[j]`, 5 fold × 3 h; Gain trên giá so với **E0** và so với **B0-306**. `Gain_E0 ≤ 0` = không có tín hiệu độc lập. Nếu có cột thắng B0-306 (`MedianGain_B0 > +ε_LGBM`) → ghi cờ đỏ (B0 bị nhiễu chi phối); không có luật riêng — R3/R4 sẽ tự thắng ở bước kiểm chứng. Model 1-feature dùng ES riêng từng fit (1 cột, rẻ). Chi phí: 306 × 15 fit tí hon trên GPU ≈ 2–4 h.

**(c) Mutual information (MI)** — `mutual_info_regression(X_j, z_h)` trên FIT của từng fold (train-only), `z_h` = target sau TargetTransform (đúng đại lượng model học), `n_neighbors = 3`, seed cố định. Null: `MI_null_j = MI(X_j, z_h xáo trộn)`. `MI − MI_null ≤ 0` (median 5 fold) = không đo được phụ thuộc. Chi phí ≈ 30 phút.

Cờ per cột (không có tier) [đã chốt]: với mỗi tiêu chí, cột được cờ **+** khi điểm số > 0 ở **ít nhất 2 trong 3 horizon** (`PI+`, `SA+`, `MI+`). Ví dụ PI > 0 ở h1, h2 nhưng < 0 ở h3 → `PI+`; PI > 0 chỉ ở h1 → không `+`. Không dùng bộ riêng theo horizon.

Bốn bộ candidate, định nghĩa thẳng bằng cờ; mỗi cột có giữ/bỏ riêng cho từng bộ:

| Bộ | Giữ cột khi | Ý nghĩa |
|---|---|---|
| R1 | `PI+` hoặc `SA+` hoặc `MI+` | chỉ bỏ cột âm cả ba tiêu chí (nhẹ) |
| R2 | `PI+` hoặc (`SA+` và `MI+`) | bỏ cột model không dùng, trừ khi có tín hiệu độc lập và phụ thuộc đo được |
| R3 | `PI+` | chỉ giữ cột model đang dùng hữu ích (mạnh) |
| R4 | `SA+` | chỉ giữ cột có tín hiệu độc lập (mạnh nhất; đây là cách xử lý trường hợp một cột đơn thắng B0-306) |

Kiểm chứng (4 run, LightGBM gốc, `15fixed_306` của phase A, seed 8586, 5 fold): mỗi bộ train → Gain trên giá so với B0-306 trên 15 ô. Chọn **B0\***: trong các bộ có `MedianGain ≥ −ε_LGBM` (không tệ hơn B0-306), lấy bộ có MedianGain cao nhất; chênh nhau < ε → lấy bộ nhỏ hơn. Không bộ nào đạt → B0\* = B0-306, ghi rõ "lọc không giúp". Chỉ MedianGain quyết định (ε_LGBM đo ở phase A trên B0-306); WinRate/P10/Worst báo cáo. B0\* là điểm xuất phát chung; sau đó mỗi model calibrate riêng trên B0\* (§1.3 phase B) rồi mới vào vòng lặp của nó.

Output: `experiments/b0_filter.csv` (306 dòng: PI per h, SA Gain vs E0 / vs B0-306 per h, MI − null per h, cờ PI+/SA+/MI+, giữ/bỏ theo R1, R2, R3, R4), kết quả 4 run kiểm chứng + bộ được chọn, bảng nhóm 38 base feature để đọc, danh sách cột B0\* đóng băng trong config.

Áp dụng cho model khác: mọi model bắt đầu Bước 2 từ B0\* (cột); LSTM dùng per phút các fine feature còn ≥ 1 cột trong B0\* (+ rv60); AutoTS base regressor = B0\*. Bộ lọc là theo LightGBM, model khác có thể đánh giá cột khác đi — chấp nhận để giữ một base chung. Tổng chi phí §1.4 ≈ 2–4 h.

---

## 2. Bước 2 — Feature selection theo từng model

Nguyên tắc: **chạy từng model một, theo thứ tự thời gian chạy tăng dần** (§2.2); **mọi model xuất phát từ cùng B0\*** (bộ tốt nhất chọn từ B0-306 và R1–R4); mỗi model calibrate riêng `15fixed_m` và ε_m trên B0\* (§1.3) rồi tự chạy cùng một vòng lặp add-one qua danh sách §2.3 theo cùng thứ tự, bằng chính model đó. Không để model nào tìm trước rồi model khác kế thừa. Kết quả: các feature set riêng F\*_LGBM, F\*_XGB, F\*_Cat, … có thể khác nhau.

### 2.1 Vòng lặp feature (áp dụng y hệt cho mỗi model `m`, xuất phát từ cùng B0\*)

Trước vòng lặp: calibrate phase B của `m` trên B0\* (§1.3) → `15fixed_m` (LightGBM/XGBoost/CatBoost) hoặc `fixed_epoch_LSTM` (LSTM), và ε_m; XGB-RF/AutoTS/TimesFM không có gì để calibrate ngoài ε_m.

`S_m := B0*`. Với từng feature `f` trong §2.3, theo đúng thứ tự:

1. Input = B0\* + các cột ext đang KEEP của `m` + `f` (giá trị tại origin t, lag 0; với LSTM/TimesFM-covariate là chuỗi theo phút của cùng cột).
2. Train `m` × 5 fold với số vòng/epoch cố định của `m` (config §2.2).
3. Metric trên giá tại VAL; Gain per ô với **base = `m` trên S_m hiện tại** (ghi thêm Gain vs `m` trên B0\*, vs E0, và Gain standalone §2.4).
4. `MedianGain ≥ −ε_m` → **KEEP** (tốt hơn hoặc gần như không đổi), `S_m := S_m + f`; `MedianGain < −ε_m` → **DROP**.
5. Feature tiếp theo.

Hết danh sách → **F\*_m** (bộ sau vòng lặp). Không còn safety-net. Sau đó:

(a) **Prune PI** (vẫn số vòng/epoch cố định): tính permutation importance trên VAL cho các cột ext của F\*_m; bỏ đồng thời mọi cột ext không có cờ PI+ (PI > 0 ở ≥ 2/3 horizon — cùng quy ước cờ §1.4; PI = median 5 fold của 3 lần xáo) → **F\*_m^prune**.

(b) **Confirmation 3 seed → win_m** (phase C): mỗi configuration (F\*_m và F\*_m^prune) chạy 3 seed (8586, 8587, 8588; ES bật; `best_iteration`/best epoch ghi lại cho Final refit) → 3 bảng RMSE 5 fold × 3 horizon = 15 ô. Với mỗi ô (f, h) lấy **mean RMSE của 3 seed** → một bảng `RMSE̅` 15 ô duy nhất cho mỗi configuration. Sau đó từng ô:

```
Gain_{f,h} = 1 − RMSE̅^prune_{f,h} / RMSE̅^unprune_{f,h}
```

rồi **MedianGain = median của 15 Gain**, so với ngưỡng nhiễu ε_m của model đang xét (WinRate/P10/Worst tính trên cùng 15 ô, chỉ báo cáo). `MedianGain ≥ −ε_m` → **win_m = F\*_m^prune**; thấp hơn → **win_m = F\*_m** (unpruned). Bảng `RMSE̅` của win_m là bảng dùng cho §3 và figure §7.3.

Kết quả của model `m`: **win_m** (feature set + bảng `RMSE̅` mean 3 seed) + bảng `keepdrop_<m>.csv` + bảng prune. Sang model kế tiếp (cũng từ B0\*). TimesFM không có feature dạng cột: xuất phát không covariate, thử thêm lần lượt candidate §2.3 làm covariate nếu API có (§2.2).

### 2.2 Thứ tự model (thời gian chạy tăng dần), config và cách chọn feature

| # | Model | Config (một config, không sweep) | Chọn feature | Ước lượng tổng cho 39 candidate (15 ngày, Vast) |
|---|---|---|---|---|
| 1 | LightGBM | `LGBMConfig` gốc (B0) | §2.1 từ B0\* với `15fixed_LGBM` → F\*_LGBM | 15 fit × ~5 s ≈ 1–2 phút/candidate → **≈ 1–1.5 h** |
| 2 | XGBoost | `hist`, `device=cuda`, `reg:pseudohubererror` (huber_slope 0.9), lr 0.03, max_depth 6, seed 8586; cùng TargetTransform | §2.1 từ B0\* với `15fixed_XGB` → F\*_XGB | **≈ 1–2 h** |
| 3 | CatBoost | GPU, `Huber:delta=0.9`, lr 0.03, depth 6, seed 8586; cùng TargetTransform | §2.1 từ B0\* với `15fixed_Cat` → F\*_Cat | **≈ 1–2 h** |
| 4 | TimesFM | zero-shot, không train; input chuỗi r1 kết thúc tại t, context 512 (hoặc tối đa API); dự báo `r̂_{t+1..t+3}` → cộng dồn → giá; pin package/checkpoint/backend; tắt tùy chọn ép dương nếu API có | (a) **TFM-POINT** không feature = baseline của model; (b) nếu API có covariate (xreg / `forecast_with_covariates` hoặc tương đương): §2.1 với candidate §2.3 làm covariate theo phút (xuất phát không covariate), giá trị cho 3 bước dự báo = giữ giá trị tại t → F\*_TFM; không có covariate thì chỉ (a); (c) nếu (a)/(b) thắng E0 (MedianGain > 0 vs E0) → **TFM-LoRA**: rank 8 trên attention/FF, train FIT, ES trên ES set, Huber trên `r̂_{t+1..t+3}`, 1 config, 3 seed | inference batch 7.2k origin ≈ 1–2 phút/run → TFM-POINT vài phút; loop covariate (nếu có) **≈ 1–1.5 h**; LoRA (nếu chạy) ≈ 1–2 h |
| 5 | XGB-RF (thay ExtraTrees) | XGBoost random-forest mode trên GPU: `num_parallel_tree=500`, `subsample=0.63`, `colsample_bynode=0.3`, `learning_rate=1`, 1 vòng boosting, `max_depth=8`, `min_child_weight=500`, squared error trên z-target, `device=cuda`, seed 8586 | §2.1 từ B0\* (1 vòng boosting cố định — không có gì để calibrate ngoài ε) → F\*_XGBRF | 15 fit × ~10 s ≈ 2–3 phút/candidate → **≈ 1.5–2 h** |
| 6 | AutoTS — 2 model cố định | `WindowRegression` (regression_model LightGBM, GPU) và `MultivariateRegression` (regression_model XGBoost, GPU) — tên chính xác và cách truyền tham số GPU cho regression_model chốt khi audit; target r1, `forecast_length = 3` → cộng dồn → giá; base regressor = B0\*; transformer cố định tối thiểu, không search, không ensemble nội bộ; số vòng của regression_model bên trong cố định trong config theo cơ chế AutoTS (không ép fixed_rounds) | **hướng 1**: mỗi model một vòng lặp §2.1 từ B0\* với candidate làm regressor → F\*_A1, F\*_A2. **Tổng hợp chỉ làm sau khi cả hai vòng lặp đã kết thúc** (F\*_A1, F\*_A2 đầy đủ): mỗi model chạy thêm đúng 1 run với `F*_A1 ∪ F*_A2`; chọn cho mỗi model bộ tốt hơn giữa {riêng, tổng hợp} theo metric | rolling-origin trên lưới thưa (mỗi 5 phút, ~1.4k origin) ≈ 3–5 phút/candidate → **≈ 2–4 h/model, 4–8 h cả hai**; run cuối chấm trên toàn bộ origin |
| 7 | LSTM-DMH | context 512; input mỗi phút = các fine feature B0 còn trong B0\* (+ rv60) + ext đang KEEP; 1 lớp LSTM hidden 64; head linear 3 output; Huber trên z-target (TargetTransform B0); Adam lr 1e-3, batch 256, ≤ 50 epoch, ES patience 5 trên ES set chỉ ở run calibrate và confirmation; seed 8586; NaN → 0 sau chuẩn hóa | §2.1 từ B0\* với `fixed_epoch_LSTM` (per fold, một số epoch cho cả 3 h), 1 seed trong vòng lặp; confirmation F\*_LSTM 3 seed, ES bật. Dự phòng nếu hết thời gian: chạy LSTM trên từng F\*_m của các model khác (4–6 run) và chọn bộ tốt nhất theo metric — không có cách biết trước bộ nào hợp | 1 fit ≈ 1–3 phút GPU cho cả 3 h; 5 fold ≈ 5–15 phút/candidate → **≈ 3–10 h** (chậm nhất → chạy cuối) |

Ràng buộc chung cho regressor/covariate (AutoTS, TimesFM): giá trị dùng để dự báo bar `s` chỉ được tính từ dữ liệu `≤ s−1`; dự báo 3 bước từ t giữ nguyên giá trị tại t (cách truyền cụ thể chốt khi audit API; kiểm tra bằng §6.4). AutoTS/TimesFM không thấy VAL/TEST. Audit version trước khi code; cài package chỉ khi user cho phép.

Tổng Bước 2 trên 15 ngày ≈ 12–25 giờ máy nếu chạy tuần tự đủ 7 model (cộng §1.4 ≈ 3–5 h vì standalone chạy GPU).

Không thêm: KNeighbors (306 chiều, low SNR → ≈ E0 hoặc noise), Bagging/ExtraTrees sklearn (CPU-only — họ bagging đại diện bởi XGB-RF trên GPU), LinearRegression/Lars (OLS trên cột collinear không ổn định), AutoTS tự search (bỏ theo quyết định user), yfinance.

### 2.3 Danh sách candidate (chung cho mọi model)

Thứ tự trong bảng = thứ tự thử. Mọi cột chỉ dùng dữ liệu ≤ t, cửa sổ kết thúc tại t, lookback ≤ 1440 phút.

Ký hiệu: `C, O, H, L, V` = close/open/high/low/volume của bar; `A` = amount (quote volume); `TP = (H + L + C)/3`; `r1 = log(C_t / C_{t−1})`; `rv_k = sqrt(mean_k(r1²))`; `EMA_k` = ewm(span k, min_periods k) trên log C; `ret_k = log(C_t / C_{t−k})`.

**A. VWAP thật từ amount** — `A/V = Σ p·q / Σ q` của các trade trong bar (Binance quote volume / base volume) → là VWAP thật, không phải proxy.

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 1 | `vwap_amt_gap_1` | `log(C / (A/V))` | vị trí close so với giá trung bình bar; gần `close_position` |
| 2 | `vwap_amt_gap_15` | `log(C / (Σ_15 A / Σ_15 V))` | |
| 3 | `vwap_amt_gap_60` | `log(C / (Σ_60 A / Σ_60 V))` | |
| 4 | `vwap_amt_gap_240` | `log(C / (Σ_240 A / Σ_240 V))` | |

**B. Return / rolling statistics ngoài B0** (B0 có ret 1, 5, 8, 32 và các lag; rv5/rv60, rv8/rv64)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 5 | `ret_60` | `log(C_t / C_{t−60})` | |
| 6 | `ret_240` | `log(C_t / C_{t−240})` | |
| 7 | `ret_1440` | `log(C_t / C_{t−1440})` | |
| 8 | `log_rv15_rv240` | `log(rv_15 / rv_240)` | |
| 9 | `log_rv60_rv1440` | `log(rv_60 / rv_1440)` | |
| 10 | `ret_skew_60` | skew của r1 trên 60 bar | |
| 11 | `dd_240` | `log(C / max_240(C))` | drawdown |
| 12 | `ru_240` | `log(C / min_240(C))` | run-up |

**C. MA / EMA / HMA** (B0 có EMA 5/20, 8/32, 16/64, 32/128, HMA 16)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 13 | `log_c_ema60` | `log C − EMA_60` | |
| 14 | `log_c_ema240` | `log C − EMA_240` | |
| 15 | `log_c_ema1440` | `log C − EMA_1440` | |
| 16 | `log_ema60_ema240` | `EMA_60 − EMA_240` | |
| 17 | `hma_slope64_volnorm` | `diff(HMA_64(log C)) / rv60` | |

**D. RSI / MACD multi-scale** (B0 có RSI 15/64; MACD 5/20/7 và 16/64/16)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 18 | `rsi240_centered` | `RSI_240(r1)/100 − 0.5` | |
| 19 | `macd_hist_60_240_60_volnorm` | `((EMA_60 − EMA_240) − EMA_60(EMA_60 − EMA_240)) / rv60` | |

**E. Bollinger** (trên log C; `SMA_n`, `σ_n` với `min_periods = n`, `ddof = 0`)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 20 | `bb_pctb_20` | `(log C − SMA_20) / (2·σ_20)` | mean-reversion z-score |
| 21 | `bb_pctb_60` | `(log C − SMA_60) / (2·σ_60)` | |
| 22 | `bb_logbw_20` | `log(σ_20)` | bandwidth |

**F. ATR / Keltner** (`TR = max(H−L, |H−C_{t−1}|, |L−C_{t−1}|)`, `ATR_n` = Wilder EMA alpha 1/n, reset sau gap)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 23 | `log_atr14_c` | `log(ATR_14 / C)` | |
| 24 | `log_atr14_rv14` | `log((ATR_14 / C) / rv_14)` | range-vol ÷ close-vol; giả thuyết liên quan bounce/reversal ở h = 1 |
| 25 | `kcw_20` | `log(2·ATR_20 / EMA_20(C))` | Keltner channel width |

**G. MFI** (money flow = A; `A⁺` khi `TP_t > TP_{t−1}`, `A⁻` khi `TP_t < TP_{t−1}`)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 26 | `mfi14_centered` | `ΣA⁺ / (ΣA⁺ + ΣA⁻)` trên 14 bar `− 0.5` | NaN nếu mẫu = 0 |
| 27 | `mfi60_centered` | như trên, 60 bar | |

**H. A/D dạng rolling** (không dùng tích lũy vì level phi dừng; `CLV = ((C−L) − (H−C)) / (H−L)`, = 0 khi `H = L`; `CLV = 2·close_position` của B0)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 28 | `ad_vwclv_5` | `Σ_5 CLV·V / Σ_5 V` | |
| 29 | `ad_vwclv_15` | `Σ_15 CLV·V / Σ_15 V` | |
| 30 | `ad_vwclv_60` | `Σ_60 CLV·V / Σ_60 V` | |

**I. Parabolic SAR** (AF 0.02, bước 0.02, max 0.2; `SAR_t` tính từ bar ≤ t; reset sau gap NaN)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 31 | `psar_dir` | +1 uptrend / −1 downtrend | |
| 32 | `psar_logdist` | `log(C / SAR_t)` | |
| 33 | `psar_age_log` | `log1p(số bar từ lần flip gần nhất)` | |

**J. Regime / calendar**

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 34 | `dow_sin`, `dow_cos` | `sin/cos(2π·weekday/7)` | thử như một cặp; trên 15 ngày chỉ có 2 tuần → ý nghĩa hạn chế |
| 35 | `log_rv60_med2d` | `log(rv60 / median_2880(rv60))` | vol regime |
| 36 | `log_range_240` | `log((max_240 H − min_240 L) / C)` | compression/breakout |

**K. Resolution 5 phút** (`BTC_lf_5min.csv`, chỉ bar đã đóng, as-of join `T ≤ t`; về thông tin trùng 1-phút, chỉ khác representation → để cuối)

| # | Cột | Định nghĩa | Ghi chú |
|---|---|---|---|
| 37 | `r5_1` | log return của bar 5' đã đóng gần nhất | |
| 38 | `r5_12` | `log(C5_T / C5_{T−12})` | |
| 39 | `log_c5_ema5_12` | `log C5 − EMA_12(log C5)` | |

Ghi chú: RMSE/MAE/R²/MAPE là metric, không phải feature. Feature ngoài danh sách chỉ thêm khi có giả thuyết rõ, và thêm vào cuối danh sách.

### 2.4 Khi candidate thua hoặc không đổi vì base đã nhiều feature

Gain so với `S_m` đo **thông tin tăng thêm** so với B0\* (đã có return/vol/candle/volume/RSI/MACD/EMA/HMA ở nhiều lag). Vì vậy phần lớn candidate sẽ ra gần 0: đó là kết luận hợp lệ ("không thêm thông tin mới"), không phải lỗi. Thua rõ (< −ε_m) nghĩa là feature thêm noise/overfit → DROP vẫn đúng. Xử lý để hiểu vì sao thua:

1. **Gain standalone (diagnostic, không dùng để KEEP/DROP)**: với mỗi candidate `f`, cùng cách tính với §1.4(b): LightGBM code gốc chỉ trên `[f]`, Gain trên giá so với E0 (và so với B0\*). Ghi vào `keepdrop_<m>.csv` cột `gain_standalone`. Đọc kết hợp: standalone > 0 nhưng vs S_m ≈ 0 → có tín hiệu nhưng trùng base; standalone ≈ 0 → không có tín hiệu. Chi phí: 15 fit tí hon ≈ vài giây/candidate.
2. **KEEP khi không đổi** (luật user) giữ lại feature bị che khuất; bước **prune PI** cuối vòng lặp (§2.1a) dọn feature thuần noise. Không có safety-net.
3. Không đổi hyperparameter của model để "giúp" feature (config cố định suốt vòng lặp).

---

## 3. Bước 3 — So sánh với champion (ngay sau mỗi model) + ensemble

- **Champion ban đầu** = LightGBM code gốc trên win_LGBM (sau vòng lặp #1 từ B0\*, prune PI, 3 seed). B0-306 và B0\* nguyên bản đều được log làm reference.
- Sau khi mỗi model `m` có **win_m** (§2.1: prune PI + 3 seed): tính từng ô `Gain_{f,h} = 1 − RMSE̅^win_{f,h} / RMSE̅^champion_{f,h}` với `RMSE̅` = bảng mean 3 seed của mỗi bên (cùng cách gộp như §2.1b) → MedianGain = median 15 Gain. `MedianGain > +ε_champion` (ε của champion đo ở §1.3) → **đổi champion** = win_m; ngược lại → **giữ champion**. Cả hai trường hợp ghi đầy đủ vào `champion_log.csv` (§7.2) và vẽ figure win vs champion (§7.3).
- TimesFM: biến thể tốt nhất trên VAL trong {TFM-POINT, F\*_TFM, TFM-LoRA} đại diện model. AutoTS: mỗi model cố định lấy bộ tốt hơn giữa {riêng, tổng hợp}.
- Latency (§7.4) ghi kèm trong `champion_log.csv` như thông tin; không phải tiêu chí đổi/giữ.
- **Ensemble** (sau model cuối): thành viên = champion + mọi model có `MedianGain vs E0 > 0` trên 15 ô (có skill thật; B0-306/B0\* là reference, không phải thành viên; TimesFM/AutoTS/LSTM là thành viên nếu đạt). (a) trung bình đều; (b) trọng số `1/MSE_VAL` (trên giá) per horizon; lấy cấu hình tốt hơn trên VAL rồi so với champion bằng đúng luật trên (`> +ε_champion` → champion = ensemble). Nếu < 2 thành viên thì không ensemble. Chọn cấu hình cuối **trước** khi chạm TEST; ghi thành viên + trọng số vào `champion_log.csv`.

---

## 4. Final evaluation (một lần, TEST 2 ngày)

- Refit **mọi** model (không chỉ champion) trên FIT `→ 01-30 23:56`, ES `01-31 00:00 → 22:56`, purge 60', dự báo toàn bộ TEST `02-01 00:00 → 02-02 21:27` (2.728 origin).
- Report per horizon trên giá: RMSE, MAE, Pearson r (thay đổi giá), directional accuracy; Gain vs B0-306, vs B0\* và vs E0; cho E0, B0-306, B0\*, từng model §2.2 tại F\*_m, ensemble. Xuất `all_models.csv`; vẽ **heatmap TEST của mọi model** và **Fig H_h của mọi model** với prediction của chúng (§7.3 Final).
- Đo latency §7.4 trên toàn bộ origin TEST cho mọi model (pass riêng, batch 1) → `latency_summary.csv`.
- Không sửa gì sau khi xem TEST. TEST chỉ được đọc bởi script final. TEST 2 ngày là kiểm tra one-shot của giai đoạn 15 ngày; TEST chính thức 30 ngày thuộc §5.

---

## 5. Để sau — data đầy đủ và scale data

Chỉ làm khi user quyết định phục hồi data (re-export không giới hạn 2 MiB):

1. Kiểm tra: 289.320 dòng, `2026-01-18 16:15 → 2026-08-07 14:14 UTC`, lưới 60 s, không dup/gap, OHLC sanity; checksum mới.
2. Dataset feature-selection 60 ngày `[05-09 13:15, 07-08 14:15)`: 5 fold × VAL 3 ngày, train region rolling 45 (FIT 40 + ES 5), purge 60'; TEST = 30 ngày cuối `07-08 14:15 → 08-07 14:11`. Kiểm tra lại B0\* và F\*_m đã chọn trên 15 ngày (regime khác).
3. Scale data từng model với F\*_m: train region 45 → 90 → 135 → full (từ 01-19 02:46), neo vào cùng VAL; chọn D\*_m = mức mà MedianGain so với mức trước `< +ε_m` hoặc full nếu vẫn cải thiện. TimesFM zero-shot không áp dụng.
4. So sánh champion + ensemble tại (F\*_m, D\*_m) → Final trên TEST 30 ngày.

---

## 6. Checklist mỗi experiment (bắt buộc, ghi vào log)

1. **Input**: checksum khớp §1.1; số dòng, khoảng thời gian, UTC, lưới 60 s, không dup, gap = 0; danh sách cột B0\* khớp config đóng băng.
2. **Target**: `y_h = log(C[t+h]/C[t])` kiểm tra tay vài origin; E0 trên VAL: RMSE giá của `P̂ = C_t` khớp `sqrt(mean((C_{t+h} − C_t)²))`.
3. **Time alignment**: partition half-open, origin cuối = `T_end − 4'`; bar 5' chỉ join khi `T ≤ t`.
4. **Leakage**: feature tính trên chuỗi cắt tại t và trên chuỗi đầy đủ phải cho cùng giá trị tại t; không `rolling(center=True)`, không shift âm; TargetTransform/scaler fit trên FIT của fold; MI §1.4 chỉ tính trên FIT; PI xáo trộn chỉ trong VAL; ES ≠ VAL; TEST chưa đọc; regressor/covariate (AutoTS, TimesFM) chỉ từ dữ liệu `≤ s−1`.
5. **Biên**: FIT/ES/VAL rời nhau; purge 60' giữa ES và VAL, và giữa train cuối và TEST.
6. **Metric**: tính trên **giá** sau decode + `exp` (`P̂ = C_t·exp(ŷ)`), không tính trên log return hay z-space; base của Gain ghi rõ (S_m / B0-306 / B0\* / E0 / champion); MedianGain trên 15 ô; AutoTS chấm trên đúng tập origin đã khai báo (thưa hay đầy đủ).
7. **Decode**: prediction qua `TargetTransform.decode` với rv60 của đúng origin (tree/LSTM) rồi `exp`; encode → decode round-trip khớp; AutoTS/TimesFM cộng dồn one-step đúng thứ tự trước khi `exp`.
8. **Hợp lý**: số vòng cố định đúng theo §1.3; `std(ŷ) ≪ std(y)` là bình thường (tín hiệu 1-phút chỉ cỡ 0.1–0.2 pp RMSE); Gain > ~1 pp vs B0/E0 → nghi leakage/bug, kiểm tra lại trước khi tin; xem figure §7.3 của vài origin để chắc prediction không lệch pha.

---

## 7. Log và visualize

Layout mẫu của mọi bảng/figure dưới đây, với **số giả**: `reports/smoke_visualize.md` (sinh bởi `reports/smoke_visualize.py`, seed 8586, không đọc data thật) — chỉ để thống nhất hình dạng output trước khi code; không phải kết quả, không trích dẫn.

### 7.1 Log mỗi run

`experiments/log.csv`: `exp_id, step, model, feature_set (danh sách cột ext), dataset_label, config_hash, seed, RMSE/MAE giá 15 ô, Gain 15 ô, MedianGain, WinRate, P10Gain, WorstGain, base, decision, ghi chú`. Thư mục `experiments/runs/<exp_id>/` chứa config, số vòng, importance, prediction VAL (ŷ và P̂ theo origin). Run phải tái tạo được từ config + seed (GPU có thể lệch bit nhỏ; ghi nhận).

### 7.2 Log quyết định

- `experiments/b0_filter.csv` (§1.4): 306 dòng — cột, base feature, lag, PI per h (median fold), SA Gain vs E0 / vs B0-306 per h, MI − null per h, cờ PI+/SA+/MI+ (> 0 ở ≥ 1 horizon), giữ/bỏ theo từng bộ R1, R2, R3, R4; kèm bảng nhóm 38 base feature và kết quả 4 run kiểm chứng + bộ được chọn.
- `experiments/keepdrop_<model>.csv`: mỗi candidate một dòng — thứ tự, cột, MedianGain/WinRate/P10/Worst vs S_m, Gain vs B0\*, Gain vs E0, `gain_standalone`, decision KEEP/DROP, size S_m sau quyết định, exp_id.
- `experiments/champion_log.csv`: mỗi model một dòng khi so với champion — model, win_m (cột ext sau prune), champion trước, metric per horizon (giá) của cả hai, bảng `RMSE̅` (mean 3 seed) của hai bên và Gain 15 ô, MedianGain/WinRate/P10/Worst, ε_champion, decision **đổi / giữ**, champion sau, exp_id. Kèm `prune_<model>.csv`: F\*_m vs F\*_m^prune (3 seed → `RMSE̅` → Gain 15 ô → MedianGain) → win_m. Ensemble và lựa chọn cuối cũng ghi vào đây.
- `experiments/summary/all_models.csv`: mọi model (E0, B0-306, B0\*, từng model tại F\*_m, TimesFM các biến thể, AutoTS ×2, LSTM, ensemble) × fold × horizon × {RMSE, MAE, r, dir-acc, Gain vs B0-306, Gain vs B0\*, Gain vs E0, Gain vs champion} + latency p95/p99/max (ms) per model × horizon (§7.4); kèm TEST riêng.

### 7.3 Visualize (theo origin, không vẽ chuỗi dự báo liên tục)

- **Màu**: actual **luôn đen**; E0 xám nét đứt. Ảnh so sánh win vs champion dùng màu theo vai trò — win = blue `#2a78d6` (▲), champion = red `#e34948` (●): cặp xa nhau nhất trong palette. Ảnh nhiều model dùng màu + marker **cố định cho từng model** (palette categorical đã validate bằng validator của skill dataviz: blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, yellow `#eda100`, magenta `#e87ba4`, green `#008300`, violet `#4a3aa7`, red `#e34948` — thứ tự cố định, không xoay vòng); tối đa 8 màu mỗi panel, vượt thì tách nhóm; reference B0-306/B0\* xám nét đứt; heatmap diverging xanh↔đỏ, cùng thang màu khi so sánh. Mapping cụ thể: `STYLE` trong `reports/smoke_visualize.py`.

**Sau mỗi model — win_m vs champion hiện tại** (2 file):
- **Fig P — forecast path** (1 ảnh = 3 panel, mỗi panel MỘT origin `t`): trục x = `t, t+1, t+2, t+3`; trục y = **thay đổi giá so với `C_t`** (USD). Trong panel: **actual** `[0, C_{t+1}−C_t, C_{t+2}−C_t, C_{t+3}−C_t]` (đen), **prediction của win** và **của champion** `[0, P̂_{t+h}−C_t]` với `P̂_{t+h} = C_t·exp(ŷ_h)`, **E0 = đường ngang 0**. Ba origin = ba ngày VAL/fold khác nhau đại diện mức biến động **thấp / trung bình / cao** (xếp 5 ngày VAL theo std r1 trong ngày, lấy min / trung vị / max), mỗi ngày lấy **origin cố định đầu tiên ≥ 12:00 UTC** — chọn theo quy tắc cố định, **không** chọn theo error/prediction.
- **Fig HM**: 2 heatmap 15 ô (fold × horizon) — của win và của champion, giá trị = Gain vs E0 tính từ bảng `RMSE̅` mean 3 seed, cùng thang màu; tiêu đề ghi MedianGain/WinRate/P10/Worst của mỗi bên và của win vs champion.
- Lưu `experiments/summary/fig_path_<model>_vs_champion.png`, `fig_HM_<model>_vs_champion.png`.

**Final (TEST)**:
- **Heatmap của mọi model** (B0-306, B0\*, mọi win_m, ensemble; một panel mỗi model, cùng thang màu): ô = khối 6 giờ × horizon (TEST 2 ngày ≈ 8 khối), giá trị Gain vs E0.
- **Fig P của mọi model**: cùng định nghĩa forecast path, vẽ prediction của **tất cả model** trên **cùng 3 origin** TEST — chọn theo std r1 của khối 60 origin không chồng nhau: thấp nhất / trung vị / cao nhất, origin đại diện = origin đầu của khối; tách 2 hàng (nhóm A: tree + ensemble; nhóm B: TimesFM/AutoTS/LSTM + reference) để mỗi panel ≤ 8 màu; actual đen ở mọi panel. Lưu `summary/fig_final_paths_all_models.png`.
- Fig D latency (§7.4) chỉ để theo dõi.
- Figure chỉ để nhìn; quyết định vẫn theo metric §0.

### 7.4 Inference latency (chỉ theo dõi — không ảnh hưởng training/loss/quyết định)

- **Đo gì**: thời gian gọi `predict` cho **một origin** (batch size 1) → ra `ŷ_h` (và `P̂`). Tree (3 model độc lập theo h): đo riêng từng horizon. Model một lần gọi ra cả 3 bước (LSTM head 3 output, TimesFM, AutoTS `fit_data + predict`): đo một lần gọi, gán chung cho h = 1, 2, 3 và đánh dấu `shared = true`. Chưa gồm thời gian tính feature (pipeline hiện tính feature theo cả frame; latency end-to-end để sau khi có pipeline incremental).
- **Đo khi nào**: pass riêng **sau khi train xong**, chỉ ở run confirmation F\*_m (§2.1c) và Final (§4), trên toàn bộ origin VAL/TEST; không đo trong vòng lặp candidate (ở đó predict theo batch cho nhanh). Pass đo không được thay đổi kết quả: assert prediction theo batch == prediction batch-1 (sai số ≤ 1e-6).
- **Cách đo**: `time.perf_counter_ns` quanh đúng lời gọi predict; model chạy GPU (LSTM/TimesFM, XGBoost nếu predict trên GPU) gọi `torch.cuda.synchronize()`/tương đương trước và sau; bỏ 50 lần gọi đầu (warm-up); số thread = mặc định thư viện (batch 1 không phụ thuộc thread; ghi cột `threads`); ghi train/predict device, phiên bản thư viện, GPU/CPU của instance.
- **Output**: `experiments/runs/<exp_id>/latency.csv` (origin, horizon, ms, shared); tóm tắt `experiments/summary/latency_summary.csv` (model × horizon × {p95, p99, max} ms, VAL và TEST riêng; p50 không cần) và cột p95/p99/max trong `all_models.csv`; ghi `train device` (luôn GPU) và `predict device` thực tế (LightGBM/CatBoost predict trên CPU là đặc tính thư viện; XGBoost/XGB-RF/LSTM/TimesFM predict GPU; AutoTS pipeline CPU quanh regression_model GPU).
- **Không dùng** latency cho KEEP/DROP, champion, ensemble hay bất kỳ quyết định nào trong plan này; chỉ ghi nhận và báo cáo.

---

## 8. Triển khai — trạng thái và lệnh chạy

**Code đã viết (2026-08-29): 74 unit test PASS (CPU, data tổng hợp), smoke end-to-end PASS, checker review đã sửa (gate, checksum §6.1, schema log).** `src/p0/`: `data` (adapter, kiểm tra §1.1 gồm gap/dup, checksum §6.1, as-of LF), `split` (§1.2), `features_ext` (39 cột §2.3, causal), `metrics` (trên giá, Gain, tóm tắt, gộp 3 seed), `transform` (tái hiện `TargetTransform` của B0 — **B0 gốc có bug nhân in-place `(n,1) *= (1,3)` trong `TargetTransform` nên không chạy được nguyên bản; file B0 không sửa, công thức giữ nguyên**), `models` (LightGBM đúng code gốc B0 qua `_make_model`/`LGBMConfig`, XGBoost, XGB-RF, CatBoost; `models_lstm`; TimesFM/AutoTS chờ researcher audit — `models_pending`), `harness` (Store = ma trận B0 + ext, `run_config`, `calibrate` → 15fixed_m, `seed_noise` → ε_m), `filter_b0` (§1.4), `loop` (§2.1 add-one, prune PI, 3 seed → win_m; §3 champion, ensemble), `latency` (§7.4), `plots` (§7.3), `logs` (§7: `log.csv`, `champion_log.csv`, `summary/latency_summary.csv` schema cố định; `runs/<exp_id>/`), `cli`. Config: `configs/p0_15d.json`. Test: `python -m pytest -q`; smoke: `python run.py smoke-e2e --out tmp_smoke --days 6` (data tổng hợp, CPU, chỉ debug). Prompt session Vast: `docs/VAST_SESSION_PROMPT.md`; bootstrap: `scripts/vast_bootstrap.sh`; môi trường: `requirements.txt`.

Ràng buộc trong CLI: (i) `calibrate / filter-b0 / loop / final` từ chối khi `.claude/MEMORY.md` còn `TRAINING: LOCKED`, và preflight GPU trước khi train; (ii) `--smoke` / `--allow-cpu` chỉ được chấp nhận khi `dataset_label` bắt đầu bằng `synthetic` — với data thật bị từ chối (cấm training CPU); (iii) mọi bước sau `check-data` verify sha256 của CSV với `data/data_checksums.json` (đã ghi cho snapshot 15 ngày, path tương đối, commit trong repo); (iv) `loop` đầu tiên phải là `lgbm` (champion ban đầu §3).

Lệnh theo bước trên Vast:

1. `bash scripts/vast_bootstrap.sh` → `python run.py check-data --config configs/p0_15d.json` (§1.1 + verify checksum; phải in `verify … OK`, 21.916 dòng, 21.258 origin, 5 fold + TEST `OK`).
2. `python run.py calibrate --model lgbm --colset b0306` (phase A: `15fixed_306`, ε_LGBM) → `python run.py filter-b0` (§1.4: PI + SA + MI → R1–R4 → 4 run kiểm chứng → `experiments/b0_star.json`).
3. `python run.py loop --model lgbm` → `xgb` → `cat` → (`tfm`, `autots_wr`, `autots_mr` sau audit) → `xgbrf` → `lstm`: mỗi lệnh = calibrate riêng trên B0\* (15fixed_m / fixed_epoch, ε_m) → 39 candidate → prune PI → 3 seed (mean RMSE từng ô) → win_m → latency → champion log + Fig H_h/HM.
4. `python run.py ensemble` (§3) → `python run.py final` (§4: TEST một lần → `summary/all_models_test.csv` gồm Gain vs E0/B0-306/B0\*/champion + latency, heatmap + Fig H_h mọi model, `latency_summary.csv`).
5. §5 khi user quyết phục hồi data đầy đủ.

## 9. Đã bỏ / đổi so với plan 2026-08-24

- HOLDOUT-NEAR/FAR → TEST 2 ngày (15 ngày) và TEST 30 ngày (data đầy đủ, để sau).
- P0 gate / canonical-pilot framework → kiểm tra §1.1; giai đoạn hiện tại chạy trên snapshot 15 ngày theo quyết định user.
- "Không ablate lại 306 feature B0" → đổi: lọc nhiễu B0 một lần bằng PI + standalone + MI + kiểm chứng (§1.4) thành B0\*; file `Baseline_LGBM.py` vẫn không sửa; B0-306 vẫn log làm reference.
- Feature dossier, Wave-1, D-family discovery, second wave → danh sách §2.3 thử lần lượt, từng model một.
- Một feature set chung → mỗi model một feature set riêng F\*_m.
- KEEP/DROP 3 vùng, daily-block paired test, confirmation framework → luật §2.1 với ε_m seed; safety-net đã bỏ, chỉ prune PI; confirmation 3 seed = mean RMSE từng ô → Gain 15 ô prune vs unprune → MedianGain ≥ −ε_m chọn prune → win_m.
- Metric trên log return → metric trên giá (USD); prediction vẫn log return.
- Scale data → để sau (§5).
- TimesFM ladder QMEAN/RECENTER/BTC-CAL → TFM-POINT (+ covariate loop nếu API có) → LoRA khi thắng E0.
- AutoTS: chỉ hướng 1 (2 model cố định, riêng/tổng hợp sau khi cả hai vòng lặp xong); hướng tự search bỏ.
- ExtraTrees (sklearn, CPU-only) → XGB-RF trên GPU [đã chốt]; training chỉ GPU, cấm CPU training.
- Số vòng cố định dùng chung từ B0-306 → mỗi model calibrate riêng `15fixed_m` trên B0\* trước vòng lặp của nó (`15fixed_306` chỉ cho lọc B0); B0\* là điểm xuất phát chung, không model nào kế thừa F\* của model khác.
- Q1–Q15, yfinance, paper 2407.18334 → bỏ.
- "True VWAP không có" → sửa: `amount/volume` là VWAP thật theo trade trong bar; chỉ biến thể từ TP·V mới cần tên `proxy`.
