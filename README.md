# P0_forecasting — BTC 1-phút point forecasting

Dự báo điểm `y_h(t) = log(C[t+h]/C[t])`, h = 1, 2, 3 phút, BTC 1-phút (Binance OHLCV + amount). Model dự báo log-return, **metric tính trên giá** (`P̂ = C_t·exp(ŷ)`, RMSE/MAE USD, Gain = 1 − RMSE_cand/RMSE_base trên 15 ô = 5 fold × 3 horizon). Chi tiết: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) (plan rev 9). Layout mẫu mọi bảng/figure với **số giả**: [`reports/smoke_visualize.md`](reports/smoke_visualize.md).

Trạng thái: **code xong (`src/p0/`, 74 unit test + smoke PASS, checker review đã sửa), chưa training** — chạy trên Vast sau khi user unlock. Giai đoạn hiện tại: snapshot 15 ngày (2026-01-18 → 02-02); raw CSV không nằm trong repo (sha256 ở `data/data_checksums.json`).

## Chạy

```bash
python -m pytest -q                                   # unit test CPU (data tổng hợp)
python run.py smoke-e2e --out tmp_smoke --days 6      # toàn bộ pipeline trên data tổng hợp, CPU, chỉ debug
python run.py check-data --config configs/p0_15d.json # kiểm tra §1.1 + verify sha256 (không training)
python run.py calibrate --model lgbm --colset b0306   # sau khi user unlock; GPU only
python run.py filter-b0 ; python run.py loop --model lgbm ; ... ; python run.py ensemble ; python run.py final
```
Bootstrap Vast: `scripts/vast_bootstrap.sh`; prompt cho session Vast: `docs/VAST_SESSION_PROMPT.md`. CLI từ chối training khi `.claude/MEMORY.md` còn `TRAINING: LOCKED`, khi CSV không khớp checksum, và từ chối `--smoke`/`--allow-cpu` trên data thật. Lưu ý: `TargetTransform` trong `Baseline_LGBM.py` có bug nhân in-place nên harness dùng bản tái hiện `src/p0/transform.py` (công thức giữ nguyên, file B0 không sửa).

## Flow tổng thể

```
Fix dataset 15 ngày (5 fold VAL 1 ngày, TEST 2 ngày cuối)
→ Lọc 306 feature B0 → B0*                                         (§1.4)
→ Mỗi model (nhanh → chậm) từ CÙNG B0*: calibrate riêng → add-one 39 candidate → F*_m   (§2)
     LightGBM → XGBoost → CatBoost → TimesFM → XGB-RF → AutoTS (2 model cố định) → LSTM
→ Sau mỗi model: prune PI → 3 seed (mean RMSE từng ô) → win_m → so với champion, log đổi/giữ, vẽ win vs champion (§3)
→ Ensemble → Final evaluation (TEST 2 ngày) → all_models.csv + figure            (§4, §7)
→ (để sau) data đầy đủ → scale data → TEST 30 ngày                                (§5)
```

## Flow lọc feature base (B0-306 → B0\*)

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

## Flow calibrate số vòng / epoch (không dùng chéo)

```
B0-306 + ES (LGBM) → 15fixed_306 → R1–R4 → B0*
   → LGBM(B0*) + ES → 15fixed_LGBM → LightGBM add-one 39 candidate → F*_LGBM
   → XGB(B0*)  + ES → 15fixed_XGB  → XGBoost  add-one 39 candidate → F*_XGB
   → Cat(B0*)  + ES → 15fixed_Cat  → CatBoost add-one 39 candidate → F*_Cat
   → LSTM(B0*) + ES theo epoch → fixed_epoch_LSTM → LSTM add-one 39 candidate → F*_LSTM
   (XGB-RF: 1 vòng boosting cố định; TimesFM: zero-shot; AutoTS: cơ chế riêng — cũng từ B0*, chỉ đo ε_m)
→ prune PI → confirmation 3 seed (mean RMSE từng ô → Gain 15 ô → MedianGain ≥ −ε_m chọn prune) → win_m → so với champion, vẽ win vs champion
```

Luật KEEP/DROP (§2.1): `MedianGain ≥ −ε_m` → KEEP (tốt hơn hoặc gần như không đổi), `< −ε_m` → DROP; ε_m = nhiễu seed của chính model đó. Chỉ MedianGain quyết định; WinRate/P10/Worst báo cáo. Training chỉ trên GPU.

## Tài liệu: chính thức / lưu trữ / tham khảo

| Loại | Vị trí | Ghi chú |
|---|---|---|
| **Chính thức** | `docs/RESEARCH_PLAN.md` | plan duy nhất có hiệu lực (rev 9) |
| Chính thức (vận hành) | `.claude/CLAUDE.md`, `.claude/MEMORY.md`, `.claude/AGENT.md`, `.claude/agents/` | hiến pháp rút gọn, trạng thái, registry agent |
| Lưu trữ (hết hiệu lực) | `docs/archive/` | plan / hiến pháp / memory bản 2026-08-24 |
| Tham khảo (không có hiệu lực) | `docs/reference/` | TimesFM R0–R6 (distribution, cũ), tổng hợp G-Research; audit API sẽ ghi vào đây |
| Layout mẫu (số giả) | `reports/smoke_visualize.md` | sinh bởi `reports/smoke_visualize.py`, không phải kết quả |

## Agents (`.claude/agents/`)

`main-controller` (điều khiển, work order, TRAINING lock, MEMORY) · `coder` (code + test tí hon) · `researcher` (audit API, giả thuyết feature, verdict methodology) · `checker` (checklist §6, review, test — phủ quyết, không sửa code) · `runner` (chạy trên Vast, chỉ khi UNLOCKED) · `analyst` (đọc log thật) · `infra` (Vast/tmux/GPU env). Chi tiết và cách phối hợp: `.claude/AGENT.md`.

## Cấu trúc repo

- `Baseline_LGBM.py` — B0 frozen (306 feature, LightGBM GPU), không sửa.
- `src/p0/` — harness theo plan §8; `run.py` CLI; `configs/`; `tests/`; `scripts/vast_bootstrap.sh`; `data/data_checksums.json`.
- `docs/` — plan chính thức + `archive/` + `reference/`.
- `reports/` — smoke visualize (số giả).
- `.claude/` — hiến pháp, trạng thái, agents, hooks.
- `data/` — manifest; CSV raw không push.
