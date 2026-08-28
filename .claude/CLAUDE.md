# P0_forecasting — Hiến pháp (bản rút gọn 2026-08-27)

File này chỉ giữ invariants. Kế hoạch nghiên cứu: `docs/RESEARCH_PLAN.md`. Trạng thái hiện tại: `.claude/MEMORY.md` (auto-import cuối file). Bản chi tiết cũ (2026-08-24) lưu ở `docs/archive/` chỉ để tham khảo, không còn hiệu lực.

Khi mâu thuẫn: quyết định user mới nhất > `docs/RESEARCH_PLAN.md` > file này > code hiện có > đề xuất. **Không tự mở rộng protocol/governance/stage/rule/framework khi user không yêu cầu.** Chỉ đổi thiết kế đã có khi phát hiện logic sai, leakage, hoặc model/metric rõ ràng không phù hợp — và nói rõ cái nào, vì sao, không thay bằng thứ phức tạp hơn.

## Bài toán

- Point forecast BTC 1 phút: `y_h(t) = log(C[t+h]/C[t])`, h ∈ {1, 2, 3}. Mọi model sinh ŷ1, ŷ2, ŷ3 trên đúng target này; model dự báo one-step return phải cộng dồn trước khi chấm. Không có distribution objective.

## Data

- Nguồn: `data\BTC_hf_1min.csv` (+ `data\BTC_lf_5min.csv` cho feature 5 phút); cột `datetime,timestamp,open,high,low,close,volume,amount`; UTC, lưới 60 s. `data\` read-only; không trộn `*_close.csv`; không tự fetch data mới khi user chưa yêu cầu.
- Giai đoạn hiện tại chạy trên **snapshot 15 ngày** đang có (file bị cắt 2 MiB: 21.916 dòng, 2026-01-18 16:15 → 02-02 21:30, dòng cuối cụt → bỏ; đã kiểm tra lưới/dup/gap/OHLC sạch). Mọi kết quả gắn nhãn dataset 15d. Phục hồi 289.320 bar và scale data để sau, theo lệnh user.
- OHLCV-only: không giả định và không đặt tên feature như order book, trades, order-flow, funding, OI… `amount/volume` là VWAP thật theo trade trong bar (Binance quote volume); biến thể volume-weighted tính từ TP·V phải mang tên `proxy`.
- Header lowercase → uppercase cho B0 xử lý ở ingestion adapter; không sửa B0.

## Baseline, model, metric, split (đã thiết kế — giữ nguyên)

- **B0 = `Baseline_LGBM.py`** frozen (settings deny Edit/Write): không sửa file, không đổi hyperparameter. 306 feature của B0 được **lọc nhiễu một lần** (plan §1.4: permutation importance + standalone 1-feature vs E0/B0-306 + mutual information; kiểm chứng bằng 3 run so với B0-306) thành **B0\*** = feature baseline chung; lọc bằng chọn cột trong harness, B0-306 vẫn log làm reference; candidate = B0\* + cột ext.
- **Model** (thứ tự chạy = thời gian chạy tăng dần): LightGBM → XGBoost → CatBoost → TimesFM (TFM-POINT; covariate loop nếu API có; thắng E0 → LoRA) → XGB-RF (XGBoost random-forest mode trên GPU, thay ExtraTrees) → AutoTS (2 model cố định, regression_model LightGBM/XGBoost GPU; tổng hợp F*_A1 ∪ F*_A2 chỉ sau khi cả hai vòng lặp xong) → LSTM-DMH (cuối); ensemble của các model tốt. **Mọi model xuất phát từ cùng B0\***; mỗi model calibrate riêng một run ES trên B0\* → `15fixed_m` (tree) / `fixed_epoch_LSTM` (LSTM) — XGB-RF/TimesFM/AutoTS theo cơ chế riêng, không ép fixed_rounds — rồi tự chạy vòng lặp add-one (KEEP nếu tốt hơn/không đổi, DROP nếu tệ hơn ε_m) → F\*_m riêng (F\*_LGBM, F\*_XGB, … có thể khác nhau); không model nào kế thừa F\* của model khác; số vòng không dùng chéo (`15fixed_306` chỉ cho lọc B0); cờ lọc B0 = > 0 ở ≥ 2/3 horizon; chạy từng model một; sau mỗi model so với champion (ban đầu = LightGBM code gốc) và log đổi/giữ. Không thêm model chỉ để dài danh sách; không AutoTS tự search.
- **Metric**: model dự báo log return ŷ_h, nhưng **metric tính trên giá** `P̂ = C_t·exp(ŷ_h)` (USD): RMSE, MAE trên `P̂_{t+h} − C_{t+h}`; `Gain = 1 − RMSE_cand/RMSE_base` per horizon × fold; MedianGain, WinRate, P10Gain, WorstGain; Pearson r và directional accuracy trên thay đổi giá `P̂ − C_t` vs `C_{t+h} − C_t`; importance chỉ là diagnostic; E0 (P̂ = C_t) luôn log. Không thêm metric mới. **Chỉ MedianGain (với ε) quyết định** KEEP/DROP, chọn B0*, champion, thành viên ensemble; WinRate/P10/Worst chỉ báo cáo; PI/MI/standalone chỉ để lập R1–R4 khi lọc B0. Inference latency (predict một origin, batch 1, p95/p99/max per model × horizon) chỉ để theo dõi — không ảnh hưởng training/loss/quyết định. Log đầy đủ mỗi candidate và mỗi lần đổi/giữ champion; bảng tổng hợp mọi model; visualize theo origin t → 3 điểm t+1..t+3 (không vẽ chuỗi dự báo liên tục).
- **Split**: walk-forward FIT → ES → purge 60' → VAL; TEST cuối chỉ chạm ở Final. 15 ngày: 5 fold VAL 1 ngày (01-27 → 01-31), expanding FIT, ES = ngày trước, TEST = 02-01 → 02-02. Data đầy đủ (để sau): 5 fold VAL 3 ngày, train region 45 (FIT 40 + ES 5), TEST 30 ngày cuối. Origin t thuộc `[T_start, T_end)` chỉ khi `t ≥ T_start` và `t + 3' < T_end`. Feature chỉ dùng τ ≤ t; TargetTransform/scaler fit train-only mỗi fold.
- Mỗi experiment đi qua checklist §6 của RESEARCH_PLAN (input, target, time alignment, leakage, biên, causality, metric, decode, hợp lý).

## Training state

`TRAINING: LOCKED` (xem MEMORY). Chỉ user unlock bằng lệnh rõ ("unlock training" / "bắt đầu training" / "run experiments"). Viết code không phải permission để train. **Training chỉ trên GPU — cấm training bằng CPU, không CPU fallback** (detect GPU tại runtime, không hard-code GPU model); ExtraTrees sklearn (CPU-only) đã thay bằng XGB-RF. CPU chỉ cho việc không phải training: tính feature, metric, MI/PI, unit/smoke test, và predict của thư viện mặc định chạy CPU (LightGBM/CatBoost). Giai đoạn 15 ngày chạy trên Vast (GPU detect tại runtime, có thể là RTX 3090, không hard-code); data đầy đủ → Vast. Vast tính giờ: mỗi run phải thuộc một bước và trả lời một câu hỏi; không idle, không chạy trùng.

## Safety & hygiene

- Không đưa secrets (Vast API key, SSH key, token, password) vào repo/CLAUDE.md/MEMORY.md/git; IP/instance id ngắn hạn không thành memory.
- Không tự ý: xóa raw data, `git reset --hard`, force push, xóa experiment results, rotate/xóa Vast resources — kể cả trong bypassPermissions.
- MEMORY.md là trạng thái, không phải log: update/replace khi obsolete; finding chỉ ghi khi đã chạy thật.
- Context: <70% bình thường; 70–85% chọn lọc, checkpoint ở ranh giới có nghĩa; ≥85% hoàn tất đơn vị nhỏ nhất → cập nhật MEMORY → `/compact`. Không compact giữa một atomic change.

@MEMORY.md
