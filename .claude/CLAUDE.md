# P0_forecasting — Hiến pháp (bản rút gọn 2026-09-03)

File này chỉ giữ invariants. Kế hoạch nghiên cứu: `docs/RESEARCH_PLAN.md` (rev 10). Code: `src/p0/` + `run.py` (§8 plan); prompt Vast: `docs/VAST_SESSION_PROMPT.md`. Trạng thái hiện tại: `.claude/MEMORY.md` (auto-import cuối file). Bản cũ ở `docs/archive/` và tài liệu tham khảo ở `docs/reference/` không có hiệu lực (trừ audit API được plan trích dẫn). Agents: `.claude/AGENT.md`.

Khi mâu thuẫn: quyết định user mới nhất > `docs/RESEARCH_PLAN.md` > file này > code hiện có > đề xuất. **Không tự mở rộng protocol/governance/stage/rule/framework khi user không yêu cầu.** Chỉ đổi thiết kế đã có khi phát hiện logic sai, leakage, hoặc model/metric rõ ràng không phù hợp — và nói rõ cái nào, vì sao, không thay bằng thứ phức tạp hơn.

## Bài toán

- Point forecast BTC 1 phút: `y_h(t) = log(C[t+h]/C[t])`, h ∈ {1, 2, 3}. Mọi model sinh ŷ1, ŷ2, ŷ3 trên đúng target này; model dự báo one-step return phải cộng dồn trước khi chấm. Không có distribution objective.

## Data

- Nguồn: OHLCV 1 phút Binance (`datetime,timestamp,open,high,low,close,volume,amount`; UTC, lưới 60 s) + bar 5 phút cho feature 5'. `data\` read-only; không trộn `*_close.csv`; không tự fetch data mới khi user chưa yêu cầu. Mỗi config trỏ đúng snapshot + file checksum sha256 riêng (§6.1) — CLI từ chối chạy khi CSV không khớp; không ghi đè file checksum khi chưa có lệnh user.
- **Vòng hiện tại = data đầy đủ (expanded)**: `configs/p0_full.json` → `data/BTC_hf_1min_full.csv` + `data/BTC_lf_5min_full.csv` (LF phải phủ toàn bộ HF), checksum `data/data_checksums_full.json`, kết quả ở `experiments/full/`. Split suy ra từ cuối data thật (§1.5 plan): TEST = 30 ngày cuối, 5 fold VAL 3 ngày liên tiếp trước TEST, train region rolling FIT 40 + ES 5 ngày, purge 60'. Không hard-code ngày. Vòng 15 ngày (2026-01-18 → 02-02) đã xong: `configs/p0_15d.json`, artifact `experiments/15d/` — lịch sử, không sửa.
- OHLCV-only: không giả định và không đặt tên feature như order book, trades, order-flow, funding, OI… `amount/volume` là VWAP thật theo trade trong bar; biến thể volume-weighted tính từ TP·V phải mang tên `proxy`.
- Header lowercase → uppercase cho B0 xử lý ở ingestion adapter; không sửa B0.

## Baseline, model, metric, split (đã thiết kế — giữ nguyên)

- **B0 = `Baseline_LGBM.py`** frozen (settings deny Edit/Write): không sửa file, không đổi hyperparameter. Bug đã biết: `TargetTransform` nhân in-place → harness dùng `src/p0/transform.py` tái hiện đúng công thức. **B0\*** (72 cột, lọc §1.4 ở vòng 15 ngày) là phần B0 của mọi S0_m; B0-306 vẫn log làm reference.
- **Điểm xuất phát vòng expanded-data (2026-09-03) = S0_m KHOÁ** = B0\* ∪ F_old_m, dựng từ `experiments/15d/wins/<m>.json` (không gõ tay); cột khoá không phải candidate, không bị prune PI, không thể bỏ. AutoTS: mỗi nhánh probe (WR/MR) kế thừa đúng bộ thắng của nhánh đó. TimesFM: S0 = ∅ (không kế thừa covariate). **Không** restart từ B0\*, **không** kế thừa số vòng/epoch/ε/RMSE/champion cũ — mọi đại lượng đo lại trên data mới.
- **Candidate = CHỈ C_short** (`src/p0/features_short.py`): feature ngắn hạn ≤ 15' trên lưới {1,2,3,4,5,8,10,15} làm dày nhất quán các họ §2.3 (VWAP gap, return, RV ratio, EMA gap, RSI, Bollinger, ATR/ATR-vs-RV, MFI, VWCLV, drawdown/run-up, range, skew, HMA slope, MACD); bỏ cửa sổ suy biến hoặc trùng B0 tại t; candidate cũ (KEEP lẫn DROP) không quay lại. `Candidate_m = C_short \ overlap(C_short, S0_m)` (trùng tên hoặc trùng giá trị; cùng indicator khác lag KHÔNG phải trùng; tương quan cao chỉ báo) — `python run.py lock-s0` ghi `experiments/<run>/s0/` (S0_m, Candidate_m, collisions.json). Không có "master feature pool" như một stage.
- **Model** (thứ tự = thời gian chạy tăng dần): LightGBM → XGBoost → CatBoost → **TimesFM-LoRA** → XGB-RF → AutoTS (WR/MR probe → `autots-search` → AutoTS-final) → LSTM; ensemble của các model tốt. Mỗi model: calibrate riêng trên S0_m bằng `calib_seed` → số vòng/epoch cố định (XGB-RF/AutoTS theo cơ chế riêng) → ε_m mới (3 `eval_seeds`) → add-one Candidate_m ở `selection_seed` (KEEP nếu ≥ −ε_m, DROP nếu tệ hơn) → F\*_raw → prune PI CHỈ cột mới → confirmation 3 seed raw vs pruned (mean RMSE từng ô, ES bật) → win_m → so champion (`> +ε_champion` đổi, ngược lại giữ). Không model nào kế thừa F\* của model khác; chạy từng model một.
- **TimesFM (2026-09-03)**: pretrained 2.5 → **LoRA fine-tune per fold** trên chuỗi r1 (FIT học, ES chọn epoch, VAL không thấy) → **freeze adapter** → cùng adapter cho toàn bộ XReg covariate add-one (thêm candidate = fit lại xreg, KHÔNG động trọng số); native LoRA lưu riêng; `tfm-final` = {LoRA + XReg(win)} nếu MedianGain vs native > +ε_TFM, ngược lại native. XReg không phải model độc lập. Chi phí LoRA ở mức fold × seed, không fold × candidate. Giữ mọi bảo vệ causality: covariate dịch 1 bar, 1 origin/lời gọi, mean head, cộng dồn one-step, xreg trên GPU (jax cuda), `PREALLOCATE=false`.
- **Metric**: model dự báo log return, **metric tính trên giá** `P̂ = C_t·exp(ŷ_h)` (USD): RMSE, MAE; `Gain = 1 − RMSE_cand/RMSE_base` per horizon × fold; MedianGain, WinRate, P10Gain, WorstGain; r và dir-acc trên thay đổi giá; E0 luôn log. **Chỉ MedianGain (với ε) quyết định** KEEP/DROP, champion, thành viên ensemble, TFM-final. Ba vai trò seed tách bạch: `calib_seed` chỉ ES; 3 `eval_seeds` đo ε (`noise = 100·std/mean` từng ô, ε = max(floor, RMS 15 ô)) và confirmation; **một `selection_seed`** cho mọi bước selection. Latency chỉ theo dõi. Log đầy đủ mỗi candidate/champion; bảng tổng hợp mọi model.
- **Fold-parallel (§9)**: 5 fold của một cấu hình chạy song song ở process riêng (`fold_workers` / `P0_FOLD_WORKERS`), kết quả y hệt tuần tự, thứ tự fold tất định, không chia sẻ state, không CPU fallback (hết VRAM → fail rõ, giảm worker). Candidate vẫn tuần tự (S đổi sau KEEP).
- **Không vẽ trong training** (§10): calibrate/loop/confirmation/champion/ensemble/final chỉ lưu artifact (origin, ŷ, metric, quyết định, PI, seed, adapter); `python run.py visualize` dựng lại mọi heatmap/path/trajectory sau khi xong, không train/inference. Định nghĩa figure §7.3 giữ nguyên: actual đen, win blue / champion red, ≤ 8 màu/panel, không vẽ chuỗi dự báo liên tục.
- **Split**: walk-forward FIT → ES → purge 60' → VAL; TEST chỉ chạm ở `final`, đúng một lần. Origin t thuộc `[T_start, T_end)` chỉ khi `t ≥ T_start` và `t + 3' < T_end`. Feature chỉ dùng τ ≤ t; TargetTransform/scaler fit train-only mỗi fold.
- Mỗi experiment đi qua checklist §6 của RESEARCH_PLAN (input, target, time alignment, leakage, biên, causality, metric, decode, LoRA, hợp lý).

## Training state

`TRAINING: LOCKED` (xem MEMORY). Chỉ user unlock bằng lệnh rõ ("unlock training" / "bắt đầu training" / "run experiments"). Viết code không phải permission để train. **Training chỉ trên GPU — cấm training bằng CPU, không CPU fallback** (detect GPU tại runtime, không hard-code GPU model). CPU chỉ cho việc không phải training: tính feature, metric, MI/PI, collision audit, unit/smoke test, visualize, và predict của thư viện mặc định chạy CPU (LightGBM/CatBoost). Chạy trên Vast (GPU detect tại runtime). Vast tính giờ: mỗi run phải thuộc một bước và trả lời một câu hỏi; không idle, không chạy trùng.

## Artifact & hygiene

- **Không đường dẫn nào dưới `experiments/**` bị gitignore** (2026-09-03): runs/, prediction, LoRA adapter (.pt + .json), log, cache đều track; nhị phân `.npz/.safetensors/.pt/.pth` dưới `experiments/` đi Git LFS (`.gitattributes`). Không commit checkpoint TimesFM gốc; ghi model id + revision + version môi trường.
- Không đưa secrets (Vast API key, SSH key, token, password) vào repo/CLAUDE.md/MEMORY.md/git; IP/instance id ngắn hạn không thành memory.
- Không tự ý: xóa raw data, `git reset --hard`, force push, xóa experiment results, rotate/xóa Vast resources — kể cả trong bypassPermissions.
- MEMORY.md là trạng thái, không phải log: update/replace khi obsolete; finding chỉ ghi khi đã chạy thật.
- Context: <70% bình thường; 70–85% chọn lọc, checkpoint ở ranh giới có nghĩa; ≥85% hoàn tất đơn vị nhỏ nhất → cập nhật MEMORY → `/compact`. Không compact giữa một atomic change.

@MEMORY.md
