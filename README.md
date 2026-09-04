# P0_forecasting — BTC 1-phút point forecasting

Dự báo điểm `y_h(t) = log(C[t+h]/C[t])`, h = 1, 2, 3 phút, BTC 1-phút (Binance OHLCV + amount). Model dự báo log-return, **metric tính trên giá** (`P̂ = C_t·exp(ŷ)`, RMSE/MAE USD, Gain = 1 − RMSE_cand/RMSE_base trên 15 ô = 5 fold × 3 horizon). Chi tiết: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) (plan rev 10.1, 2026-09-04).

Trạng thái: **vòng 15 ngày đã chạy xong (2026-09-01, artifact `experiments/15d/`, champion xgbrf, tín hiệu ≈ 0 — xem `.claude/MEMORY.md`)**. **Vòng expanded-data: code/config/doc đã migrate (2026-09-03) + pass hiệu chỉnh (2026-09-04), 139 unit test + smoke PASS, `TRAINING: LOCKED`** — chạy trên Vast khi user unlock và data đầy đủ đã đặt lên đĩa. Raw CSV không nằm trong repo (sha256 ở `data/data_checksums*.json`).

## Chạy

```bash
python -m pytest -q                                        # unit test CPU (data tổng hợp, stub cho timesfm/autots)
python run.py smoke-e2e --out tmp_smoke --days 6           # toàn bộ pipeline trên data tổng hợp, CPU, chỉ debug
python run.py check-data --config configs/p0_full.json --write-checksums   # data đầy đủ: kiểm tra §1.1 + split rolling §1.5 + anchor sha256
python run.py lock-s0   --config configs/p0_full.json      # S0_m khoá toàn bộ + overlap audit per model + Candidate_m → experiments/full/s0/ (không training)
python run.py loop --config configs/p0_full.json --model lgbm   # sau khi user unlock; GPU only; rồi xgb → cat → tfm → tfm-final → xgbrf
                                                                 # → autots_wr → autots_mr → autots-search → lstm → ensemble → final
python run.py visualize --config configs/p0_full.json      # hậu kỳ: mọi figure từ artifact, không train/inference
```
Bootstrap Vast: `scripts/vast_bootstrap.sh`; prompt cho session Vast: `docs/VAST_SESSION_PROMPT.md`. CLI từ chối training khi `.claude/MEMORY.md` còn `TRAINING: LOCKED`, khi CSV không khớp checksum, khi LF 5' không phủ HF, và từ chối `--smoke`/`--allow-cpu` trên data thật. `TargetTransform` trong `Baseline_LGBM.py` có bug nhân in-place nên harness dùng bản tái hiện `src/p0/transform.py` (công thức giữ nguyên, file B0 không sửa).

## Flow vòng expanded-data (2026-09-03, hiệu chỉnh 2026-09-04)

```
Data đầy đủ → split rolling neo vào cuối data (5 fold VAL 3 ngày, FIT 40 + ES 5, TEST 30 ngày cuối)     (§1.5)
→ S0_m = B0* ∪ F_old_m KHOÁ TOÀN BỘ (locked_b0/locked_ext, từ experiments/15d/wins/<m>.json); C_short 163 feature ≤ 15'
  → Candidate_m = C_short \ overlap(C_short, S0_m) RIÊNG từng model (chỉ trùng tên/giá trị với S0_m; tương quan chỉ báo)   (§0b, §2.3b)
→ Mỗi model từ S0_m của CHÍNH nó: calibrate trên data mới (số vòng/epoch + ε mới) → add-one Candidate_m → F*_raw
  → prune PI CHỈ cột mới → confirmation 3 seed raw vs pruned (mean RMSE từng ô) → win_m → so champion (> +ε)       (§2, §3)
     LightGBM → XGBoost → CatBoost → TimesFM-LoRA → XGB-RF → AutoTS (WR/MR probe → autots-search) → LSTM
→ Ensemble → Final (TEST một lần, lưu final/*.npz) → visualize hậu kỳ                                        (§3, §4, §7.5)
```

- **TimesFM-LoRA** (§2.2 #4): pretrained 2.5 → calibrate = LoRA fine-tune trên FIT + ES chọn epoch → freeze adapter → baseline **TimesFM-LoRA native** (không B0*) → XReg add-one trên CÙNG adapter (thêm candidate = fit lại xreg, không động trọng số) → `tfm-final`: {TimesFM-LoRA + XReg(F_best)} vs {TimesFM-LoRA native} theo luật project. Artifact `wins/tfm_lora_native.json`, `wins/tfm_lora_xreg.json`, `wins/tfm.json` (metadata LoRA/native/covariates). Audit: `docs/reference/audit_timesfm_lora.md`.
- **Fold-parallel** (§0b.6): `P0_FOLD_WORKERS=5` / `fold_workers` — 5 fold chạy song song, kết quả y hệt tuần tự, không CPU fallback.
- **Bất biến cứng, không tương tác**: GPU-only (preflight thật, không CPU fallback), TEST đúng một lần (`final/TEST_SENTINEL.json`), checksum/biên/S0 — vi phạm → `checker_log.hard_fail` (ERROR + dừng). Checker không tương tác: finding vào `experiments/<run>/checker_log.jsonl` (`scripts/checker_record.py`); ERROR chặn run, WARN/INFO ghi rồi tiếp tục.
- **Artifact**: `experiments/**` không bao giờ bị gitignore; `.npz/.pt/.png` đi Git LFS; không commit checkpoint TimesFM gốc.

## Vòng 15 ngày (lịch sử, đã xong)

Flow cũ: lọc 306 feature B0 → B0\* (= R4, 72 cột) → mỗi model từ B0\* add-one 39 candidate §2.3 → prune → confirmation → champion → ensemble → Final TEST 2 ngày. Kết quả và figure: `experiments/15d/` + `.claude/MEMORY.md` "Experiment Findings". Config: `configs/p0_15d.json`. Feature thắng của vòng này chính là phần khoá của S0_m.

Luật KEEP/DROP (§2.1): `MedianGain ≥ −ε_m` → KEEP, `< −ε_m` → DROP; ε_m = nhiễu seed của chính model đó trên data hiện tại. Chỉ MedianGain quyết định; WinRate/P10/Worst báo cáo. Training chỉ trên GPU.

## Tài liệu: chính thức / lưu trữ / tham khảo

| Loại | Vị trí | Ghi chú |
|---|---|---|
| **Chính thức** | `docs/RESEARCH_PLAN.md` | plan duy nhất có hiệu lực (rev 10.1) |
| Chính thức (vận hành) | `.claude/CLAUDE.md`, `.claude/MEMORY.md`, `.claude/AGENT.md`, `.claude/agents/` | hiến pháp rút gọn, trạng thái, registry agent |
| Audit API (căn cứ code) | `docs/reference/audit_timesfm_lora.md`, `audit_timesfm.md`, `audit_autots.md` | version/API đã kiểm; phần khác của `docs/reference/` chỉ tham khảo |
| Lưu trữ (hết hiệu lực) | `docs/archive/` | plan / hiến pháp / memory bản 2026-08-24 |
| Layout mẫu (số giả) | `reports/smoke_visualize.md` | sinh bởi `reports/smoke_visualize.py`, không phải kết quả |

## Agents (`.claude/agents/`)

Bốn agent: `checker` (verify độc lập checklist §6 + review code, KHÔNG tương tác: finding → `checker_log.jsonl`, ERROR chặn run) · `researcher` (audit API/version trước khi code, trọng tài methodology) · `analyst` (sau full run: đọc kết quả thật, anomaly/failure/regime, đề xuất) · `infra` (GPU/env troubleshooting trên Vast). Chi tiết: `.claude/AGENT.md`.

## Cấu trúc repo

- `Baseline_LGBM.py` — B0 frozen (306 feature, LightGBM GPU), không sửa.
- `src/p0/` — harness theo plan §8 (`features_short`, `s0`, `fold_parallel`, `lora`, `models_tfm`, `visualize`, `cli`, …); `run.py` CLI; `configs/`; `tests/`; `scripts/`.
- `experiments/15d/` (vòng 15 ngày) · `experiments/full/` (vòng expanded-data: `s0/`, `checker_log.jsonl` đã có; còn lại sinh khi chạy).
- `docs/` — plan chính thức + `reference/` (audit) + `archive/`. `reports/` — smoke visualize (số giả). `.claude/` — hiến pháp, trạng thái, agents, hooks. `data/` — manifest + checksum; CSV raw không push.
