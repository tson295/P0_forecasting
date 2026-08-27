> **ARCHIVED 2026-08-27 — không còn hiệu lực.** Bản này được thay bằng plan 4 bước đơn giản hóa trong `docs/RESEARCH_PLAN.md` / `.claude/CLAUDE.md`. Giữ chỉ để tham khảo lịch sử.

# P0_forecasting — Research Constitution

File này là **hiến pháp nghiên cứu**: chỉ chứa invariants ổn định. Trạng thái tiến hóa nằm ở @MEMORY.md (auto-import). Roadmap thí nghiệm chi tiết: `docs/RESEARCH_PLAN.md`. Registry agent: `.claude/AGENT.md`.

Precedence khi có mâu thuẫn:
`explicit latest user decision > older user decision > research plan > existing implementation > proposal/guess`.
Luôn phân biệt: **FROZEN USER DECISION / CURRENT IMPLEMENTATION / CURRENT PROPOSAL / FUTURE EXPERIMENT**. Không biến proposal thành fact.

## Problem

Dự báo short-horizon return của BTC trên nến 1 phút với **một objective canonical duy nhất: POINT FORECASTING**. Mọi model cuối cùng phải sinh ba scalar prediction \(\hat y_1,\hat y_2,\hat y_3\). Quantile/path machinery, nếu có trong TimesFM, chỉ là internal mechanism để tạo point estimate; distribution forecasting không còn là track/objective độc lập.

## Data contract — OHLCV-only

- Intended canonical source: `data\*_hf_1min.csv`, `data\*_lf_5min.csv` (cột `datetime,timestamp,open,high,low,close,volume,amount`). `data\` là **read-only**. KHÔNG trộn với `*_close.csv`.
- Manifest mô tả BTC 1-phút 289.320 bar, 2026-01-18 16:15 → 2026-08-07 14:14 UTC, nhưng audit 2026-08-24 cho thấy local snapshot bị truncate (BTC chỉ 21.917 data rows; ETH/SOL/XRP cũng thiếu lớn). Manifest là intended contract, không phải bằng chứng bytes local đầy đủ.
- **P0 DATA-INTEGRITY GATE**: cấm real experiment cho tới khi canonical CSV được phục hồi; row count/range/UTC/grid/gap/schema được verify; snapshot được freeze; `data\data_checksums.json` được tạo cho bản đã xác minh. File checksum hiện chưa tồn tại; không tạo checksum “canonical” từ snapshot truncate.
- Cross-asset OHLCV (ETH/SOL/XRP; HYPE 1-min chỉ có 4,5 ngày) được phép — vẫn thuộc information set thực tế. **Không fetch data mới** nếu user chưa yêu cầu.
- KHÔNG tồn tại và KHÔNG được giả định: order book, book ticker, aggregate trades, order-flow/bid-ask imbalance, spread/depth/microprice, funding, liquidation, open interest, futures positioning, mọi microstructure feature không suy ra hợp lệ từ OHLCV. Không tạo feature giả rồi đặt tên như các quantity đó. True VWAP không có — mọi biến thể volume-weighted price phải mang tên `proxy`.

## Target contract

```
y_h(t) = log(C[t+h] / C[t]),  h ∈ {1, 2, 3} phút
```
- Mọi candidate (tree, TimesFM native/custom/LoRA, LSTM) phải được quy về scalar y_1, y_2, y_3.
- TimesFM nếu forecast one-step returns r[t+1..t+3] thì PHẢI reconstruct: y_h = Σ_{i=1..h} r[t+i]; C[t+h] = C[t]·exp(y_h).
- INVARIANT (có test/assert): mọi model được đánh giá trên đúng cùng y_h — không bao giờ so one-step return của model này với cumulative h-step return của model khác.
- Với mọi half-open partition [T_start,T_end), origin t chỉ eligible nếu toàn bộ target ở trong partition: t+h_max·60s < T_end, h_max=3. Không chỉ assert t<T_end.

## MODEL SCOPE (invariant)

```
Core trees      : frozen LightGBM B0; LightGBM candidate/B*; XGBoost; CatBoost; eligible tree point ensemble
Foundation TS   : TimesFM native point; point-oriented custom variants; LoRA/adaptation nếu API hỗ trợ
DL exception    : đúng một LSTM research track, proposal LSTM-DMH-512
```
LSTM exception không mở GRU, TCN, generic Transformer, PatchTST, MLP sequence model, Chronos, TimeGPT hoặc DL family khác. Không tự thêm RF/ET/SGD hoặc model từ AutoTS/paper. yfinance không phải model; AutoTS chỉ có thể là constrained auxiliary harness.

## Research direction

- **Feature/tree branch**: TRAINING_LOCKED literature/indicator discovery → causal/redundancy review → freeze finite Wave-1 (X1–X5 chỉ là seeds; D1–Dk được phép) → sequential ablation → confirmation → data scaling → LightGBM/XGBoost/CatBoost comparison → optional tree point ensemble.
- **TimesFM branch**: TFM-POINT native benchmark bắt buộc; legacy R0→R6 được map/refactor theo khả năng thay point estimate. Quantile/path machinery phải reconstruct cumulative target trước point extraction và chỉ chấm bằng point metrics.
- **LSTM branch**: LSTM-DMH-512 là một research proposal, fair/base information set trước, B* chỉ ở explicit follow-up; không hyperparameter sweep lớn và không probabilistic objective.
- **Auxiliary references**: yfinance/AutoTS/paper chỉ đúng vai trò ghi trong roadmap; không tự đổi data/model/metric contract.
- **TimesFM version discipline**: mọi claim/kết quả gắn package/checkpoint/backend/API cụ thể; version audit trước implementation, không load checkpoint nặng khi TRAINING_LOCKED.

## Frozen decisions

1. **B0 = `Baseline_LGBM.py`** (root) — frozen: không sửa file, không ablate lại feature đã có trong B0, hyperparams đóng băng suốt ablation. (Settings đã deny Edit/Write file này.)
2. **Objective point-only**: mọi survivor sinh \(\hat y_1,\hat y_2,\hat y_3\); TimesFM distribution machinery chỉ phục vụ point extraction.
3. **Final holdouts = HOLDOUT-NEAR + HOLDOUT-FAR**, cùng one-shot stage sau khi lock mọi quyết định. Exact timestamps/Far gap còn [PLAN]; không dùng Near để sửa model trước Far.
4. **Metric contract**: Gain per horizon×fold so với frozen point baseline đã đăng ký (ban đầu B0), MedianGain, WinRate, P10Gain/WorstGain; secondary RMSE, MAE, Pearson r, directional accuracy, existing importance/diagnostics. E0 luôn log.
5. **No-new-metric**: không tự thêm CRPS/pinball/coverage/sharpness/energy/variogram/R²/MAPE/sMAPE/Sharpe/IC/calibration slope/variance ratio hoặc metric canonical khác. Feature/hypothesis được auto-discover; metric không.
6. **Validation**: walk-forward 5 folds, VAL 3 ngày, rolling 45-day initial budget, ES = 5 ngày cuối train-region, purge 60 phút, initial dev budget 60 ngày nếu restored range hỗ trợ. ES không phải VAL; B0 thực tế monitor Huber. Mọi transform/calibration fit train-only.
7. **Partition availability**: toàn bộ y_h của một origin phải nằm trong chính FIT/ES/VAL/Near/Far partition; không được mượn target qua boundary.
8. **Feature discovery**: X1–X5 là seed, không phải whitelist; finite D1–Dk Wave-1 phải được research, document, causal/redundancy review và freeze trước real VAL.
9. **Significance/reproducibility**: target h=2,3 chồng lấp nên không coi per-bar errors iid; daily blocks đã có được giữ. Seed deterministic, run tái tạo từ config và log schema trong roadmap.

## Training state

```
TRAINING_LOCKED = true
TRAINING_TARGET = VAST.AI_GPU
REAL_EXPERIMENT_CPU_FALLBACK = false
```
- Chưa chạy training/experiment thật. Phase hiện tại: governance update / data-integrity planning / feature research / architecture / static verification / unit test / leakage test / remote infra prep.
- Viết trainer/config/setup script/GPU detection KHÔNG phải permission để training. Chỉ unlock khi user nói rõ đại ý "bắt đầu training / run experiments / unlock training"; dù unlock, P0 data-integrity gate vẫn phải PASS trước real fit. Transition theo `docs/RESEARCH_PLAN.md` §12.
- Raw CSV lowercase → B0 uppercase là implementation blocker; xử lý bằng canonical ingestion/schema adapter có test, không sửa B0.
- Không load TimesFM checkpoint nặng, không chạy folds/holdouts, không tải data và không cài package trong TRAINING_LOCKED.
- Real experiment: GPU required or FAIL LOUDLY. CPU chỉ cho unit/synthetic/smoke test không-training. Không hard-code GPU model (P100/T4/A100/4090/H100…) — detect tại runtime; config `require_p100` trong B0 phải được override generic khi chạy thật.
- Vast tính tiền theo giờ: trước mỗi compute-heavy run phải trả lời được *"run này trả lời câu hỏi gì?"*; không idle GPU, không duplicate run, không bỏ checkpoint hợp lệ.

## Safety & hygiene

- Không đưa secrets (Vast API key, SSH key, token, password) vào repo/CLAUDE.md/MEMORY.md/git; IP/instance id ngắn hạn không thành durable memory.
- Không tự ý: xóa raw data, `git reset --hard`, force push, xóa experiment results, rotate/delete Vast resources — kể cả trong bypassPermissions.
- Memory hygiene: MEMORY.md là trạng thái, không phải log — update/replace khi obsolete, không append mâu thuẫn, không dump chat/code. Finding chỉ được ghi khi đã chạy thật.
- Context policy: <70% làm việc bình thường; 70–85% chọn lọc context, checkpoint ở ranh giới có nghĩa; ≥85% hoàn tất đơn vị an toàn nhỏ nhất → cập nhật MEMORY (state, files changed, decisions, unresolved, exact next step) → chủ động chạy `/compact`. Không compact giữa một atomic change.

@MEMORY.md
