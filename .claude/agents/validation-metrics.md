---
name: validation-metrics
description: Implement và kiểm định metric trên giá, Gain 15 ô, fold 15 ngày, ε/số vòng cố định, log all_models/champion_log và latency cho P0_forecasting theo §0, §1.2, §1.3, §7 của docs/RESEARCH_PLAN.md.
model: inherit
effort: high
---

Spec: `docs/RESEARCH_PLAN.md` §0 (metric), §1.2 (fold), §1.3 (ε, số vòng), §3 (champion), §7 (log, figure, latency).

Metric (không thêm mới):
- Prediction log-return `ŷ_h` → `P̂_{t+h} = C_t·exp(ŷ_h)`; `e_h = P̂_{t+h} − C_{t+h}` (USD). RMSE, MAE trên `e_h`.
- `Gain = 1 − RMSE_cand/RMSE_base` per horizon × fold (pp); MedianGain, WinRate, P10Gain, WorstGain trên 15 ô. Base phải ghi rõ (S_m / B0-306 / B0* / E0 / champion).
- Pearson r và directional accuracy trên `P̂ − C_t` vs `C_{t+h} − C_t`; dir-acc bỏ bar `C_{t+h} = C_t`. E0 (`P̂ = C_t`) luôn log.

Fold 15 ngày (§1.2): expanding FIT từ 01-19 02:46; ES = ngày trước VAL (00:00 → 22:56); purge 60'; VAL 1 ngày × 5 (01-27 → 01-31); TEST 02-01 00:00 → 02-02 21:27. Origin t thuộc `[T_start, T_end)` chỉ khi `t + 3' < T_end`. Data đầy đủ (§5, để sau): 5 fold VAL 3 ngày, train region 45.

ε_m và số vòng (§1.3): 3 seed trên feature baseline → 30 Gain → `ε_m = max(0.005 pp, std)`; best_iteration từ run baseline có ES, cố định cho candidate.

Log (§7): `log.csv`, `keepdrop_<model>.csv`, `champion_log.csv` (đổi/giữ đều ghi), `summary/all_models.csv` (+ latency p50/p95/p99), `latency_summary.csv`. Layout mẫu (số giả): `reports/smoke_visualize.md`.

Mọi hàm metric deterministic, có unit test (E0 sanity, decode → giá, Gain 15 ô). TRAINING_LOCKED: chỉ synthetic/unit; không chạy fold thật.
