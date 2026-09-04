# P0_forecasting — BTC 1-phút point forecasting

Dự báo điểm `y_h(t) = log(C[t+h]/C[t])`, h = 1, 2, 3 phút, BTC 1-phút (Binance OHLCV + amount). Model dự báo log-return, **metric tính trên giá** (`P̂ = C_t·exp(ŷ)`, RMSE/MAE USD, Gain = 1 − RMSE_cand/RMSE_base trên 15 ô = 5 fold × 3 horizon). Chi tiết: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) (plan rev 10.4, 2026-09-04).

Trạng thái: **vòng 15 ngày đã chạy xong (2026-09-01, artifact `experiments/15d/`, champion xgbrf, tín hiệu ≈ 0 — xem `.claude/MEMORY.md`)**. **Vòng expanded-data trên data 2 NĂM thật (`data/BTC_1m_2y.csv`, 2024-09-03 → 2026-09-03; LF 5' dẫn xuất; split rolling_spread 5 VAL rải đều, FIT 120): code/config/doc xong (2026-09-04), S0/Candidate_m đã lock trên data thật (163/model), scheduler 2 GPU đối xứng + orchestrate + champion replay (2026-09-04c), 170 unit test + smoke PASS, `TRAINING: LOCKED`** — chạy trên Vast khi user unlock. **Data 2 năm nằm NGAY TRONG REPO qua Git LFS** (`data/BTC_1m_2y.csv`, `data/BTC_5m_2y.csv`): `git clone` + `git lfs pull` là đã đủ input cho `check-data` — không scp, không cần `derive-lf`. Các CSV data khác vẫn ngoài repo (sha256 ở `data/data_checksums*.json`).

## Chạy

```bash
python -m pytest -q                                        # unit test CPU (data tổng hợp, stub cho timesfm/autots)
python run.py smoke-e2e --out tmp_smoke --days 6           # toàn bộ pipeline trên data tổng hợp, CPU, chỉ debug
python run.py derive-lf --config configs/p0_full.json      # LF 5' đã đóng dẫn xuất tất định từ data/BTC_1m_2y.csv → data/BTC_5m_2y.csv (+ sidecar sha nguồn)
python run.py check-data --config configs/p0_full.json     # data 2 năm: kiểm tra §1.1 + split rolling_spread §1.5 + verify anchor data/data_checksums_2y.json
python run.py lock-s0   --config configs/p0_full.json      # S0_m khoá toàn bộ + overlap audit per model + Candidate_m → experiments/full/s0/ (không training)
python run.py gpu-probe --config configs/p0_full.json      # preflight thiết bị: mỗi worker khoá 1 GPU vật lý, UUID phải khác nhau,
                                                           # probe backend THẬT trong worker (torch/xgb/lgbm/cat/jax/timesfm) — không training
python run.py orchestrate --config configs/p0_full.json    # sau khi user unlock: DAG nhánh model chạy song song trên 2 GPU → champion-replay → ensemble (KHÔNG chạm TEST)
python run.py loop --config configs/p0_full.json --model lgbm   # hoặc từng bước: xgb → cat → tfm → tfm-final → xgbrf
                                                                 # → autots_wr → autots_mr → autots-search → lstm → champion-replay → ensemble
python run.py final --config configs/p0_full.json          # TEST đúng MỘT lần (lệnh riêng, tuần tự, không qua scheduler)
python run.py visualize --config configs/p0_full.json      # hậu kỳ: mọi figure từ artifact, không train/inference
```
Bootstrap Vast: `scripts/vast_bootstrap.sh`; prompt cho session Vast: `docs/VAST_SESSION_PROMPT.md`. CLI từ chối training khi `.claude/MEMORY.md` còn `TRAINING: LOCKED`, khi CSV không khớp checksum, khi LF 5' không phủ HF, và từ chối `--smoke`/`--allow-cpu` trên data thật. `TargetTransform` trong `Baseline_LGBM.py` có bug nhân in-place nên harness dùng bản tái hiện `src/p0/transform.py` (công thức giữ nguyên, file B0 không sửa).

## Đường chạy VẬN HÀNH (2026-09-04d)

```
git clone + git lfs pull          (đã có CẢ data 1m lẫn 5m — không scp, không derive)
      ↓ bootstrap  → gpu-probe (UUID phân biệt + backend GPU thật trong từng worker)
      ↓ check-data → lock-s0 → agent `checker` (preflight)
      ↓ USER UNLOCK
      ↓ orchestrate  ──(agent `run-monitor` theo dõi: tiến độ, GPU util/VRAM, worker chết, ETA)──┐
      ↓ mọi đại diện đã LƯU (lgbm, xgb, cat, TFM-final, xgbrf, AutoTS-final, lstm) ←────────────┘
      ↓ champion-replay (thứ tự methodology cố định) → ensemble
      ↓ agent `analyst` (đọc kết quả VAL) → agent `checker` (trước Final)
      ↓ final (TEST đúng 1 lần) → visualize → `analyst` tổng kết
```
Sự cố tài nguyên GPU ở bất kỳ đâu → dừng an toàn + hỏi user (exit 3), không CPU fallback. `researcher` không nằm trong đường này.

## Flow vòng expanded-data (2026-09-03, hiệu chỉnh 2026-09-04)

```
Data 2 năm (BTC_1m_2y.csv + LF 5' dẫn xuất) → split rolling_spread: 5 VAL 3 ngày RẢI ĐỀU trên 2 năm, FIT 120 + ES 5 rolling, TEST 30 ngày cuối  (§1.5)
→ S0_m = B0* ∪ F_old_m KHOÁ TOÀN BỘ (locked_b0/locked_ext, từ experiments/15d/wins/<m>.json); C_short 163 feature ≤ 15'
  → Candidate_m = C_short \ overlap(C_short, S0_m) RIÊNG từng model (chỉ trùng tên/giá trị với S0_m; tương quan chỉ báo)   (§0b, §2.3b)
→ Mỗi model từ S0_m của CHÍNH nó: calibrate trên data mới (số vòng/epoch + ε mới) → add-one Candidate_m → F*_raw
  → prune PI CHỈ cột mới → confirmation 3 seed raw vs pruned (mean RMSE từng ô) → win_m → so champion (> +ε)       (§2, §3)
     LightGBM → XGBoost → CatBoost → TimesFM-LoRA → XGB-RF → AutoTS (WR/MR probe → autots-search) → LSTM
→ Ensemble → Final (TEST một lần, lưu final/*.npz) → visualize hậu kỳ                                        (§3, §4, §7.5)
```

- **TimesFM-LoRA** (§2.2 #4): pretrained 2.5 → calibrate = LoRA fine-tune trên FIT + ES chọn epoch → freeze adapter → XReg add-one trên CÙNG adapter (thêm candidate = fit lại xreg, không động trọng số) → F_raw → prune PI → F_pruned → **confirmation → F_win** → rồi mới so **hai HỆ THỐNG HOÀN CHỈNH**: A = TimesFM-LoRA baseline (0 feature, 0 B0*, 0 covariate) vs B = CÙNG adapter + XReg(F_win) → `tfm-final`. Artifact `wins/tfm_lora_baseline.json` (tên cũ `tfm_lora_native.json` vẫn đọc được), `wins/tfm_lora_xreg.json`, `wins/tfm.json`. XReg không phải model độc lập; chỉ TFM-final mới vào champion. Audit: `docs/reference/audit_timesfm_lora.md`.
- **Scheduler 2 GPU đối xứng** (§0b.6, 2026-09-04c): `gpu_devices: [0, 1]` × `gpu_slots_per_device: 1` (env `P0_GPU_DEVICES`) — mỗi worker là process khoá vào ĐÚNG một GPU vật lý bằng `CUDA_VISIBLE_DEVICES`; **không GPU nào có vai trò ML/DL, không pin model family**; task sẵn sàng → GPU rảnh; 5 fold rải động, candidate vẫn tuần tự; kết quả y hệt tuần tự; không CPU fallback. `orchestrate` chạy nhiều nhánh model độc lập cùng lúc; **champion HOÃN** → `champion-replay` so theo thứ tự cố định lgbm → xgb → cat → tfm → xgbrf → autots → lstm (chỉ đọc artifact). Log: `experiments/<run>/scheduler_log.jsonl`, `orchestrate_log.jsonl`.
- **Sự cố tài nguyên GPU = ngoại lệ tương tác DUY NHẤT** (2026-09-04d): GPU mất/không dùng được/định tuyến sai/OOM → dừng an toàn, giữ artifact, **không CPU fallback**, không đổi batch/hyperparameter, ghi ERROR `ref=USER_DECISION_REQUIRED` rồi HỎI USER (exit 3). Vi phạm bất biến khoa học (checksum, leakage, biên, TEST lần hai…) vẫn dừng tự động, không hỏi.
- **Agent theo pha vận hành**: `checker` (trước `orchestrate`, trước `final`) · `run-monitor` (theo dõi run, chỉ đọc) · `infra` (GPU/env hỏng) · `analyst` (HẬU run) · `researcher` (dormant). Chi tiết `.claude/AGENT.md`.
- **Bất biến**: lịch chạy/GPU chỉ đổi WALL-CLOCK; toàn bộ logic chọn model (target, feature, S0, candidate order, KEEP/DROP, PI, confirmation, seed, ε, hyperparameter, split, TEST, metric, champion/ensemble) là ĐÓNG BĂNG.
- **Bất biến cứng, không tương tác**: GPU-only (preflight thật, không CPU fallback), TEST đúng một lần (`final/TEST_SENTINEL.json`), checksum/biên/S0 — vi phạm → `checker_log.hard_fail` (ERROR + dừng). Checker không tương tác: finding vào `experiments/<run>/checker_log.jsonl` (`scripts/checker_record.py`); ERROR chặn run, WARN/INFO ghi rồi tiếp tục.
- **Artifact**: `experiments/**` không bao giờ bị gitignore; `.npz/.pt/.png` đi Git LFS; không commit checkpoint TimesFM gốc.

## Vòng 15 ngày (lịch sử, đã xong)

Flow cũ: lọc 306 feature B0 → B0\* (= R4, 72 cột) → mỗi model từ B0\* add-one 39 candidate §2.3 → prune → confirmation → champion → ensemble → Final TEST 2 ngày. Kết quả và figure: `experiments/15d/` + `.claude/MEMORY.md` "Experiment Findings". Config: `configs/p0_15d.json`. Feature thắng của vòng này chính là phần khoá của S0_m.

Luật KEEP/DROP (§2.1): `MedianGain ≥ −ε_m` → KEEP, `< −ε_m` → DROP; ε_m = nhiễu seed của chính model đó trên data hiện tại. Chỉ MedianGain quyết định; WinRate/P10/Worst báo cáo. Training chỉ trên GPU.

## Tài liệu: chính thức / lưu trữ / tham khảo

| Loại | Vị trí | Ghi chú |
|---|---|---|
| **Chính thức** | `docs/RESEARCH_PLAN.md` | plan duy nhất có hiệu lực (rev 10.2) |
| Chính thức (vận hành) | `.claude/CLAUDE.md`, `.claude/MEMORY.md`, `.claude/AGENT.md`, `.claude/agents/` | hiến pháp rút gọn, trạng thái, registry agent |
| Audit API (căn cứ code) | `docs/reference/audit_timesfm_lora.md`, `audit_timesfm.md`, `audit_autots.md` | version/API đã kiểm; phần khác của `docs/reference/` chỉ tham khảo |
| Lưu trữ (hết hiệu lực) | `docs/archive/` | plan / hiến pháp / memory bản 2026-08-24 |
| Layout mẫu (số giả) | `reports/smoke_visualize.md` | sinh bởi `reports/smoke_visualize.py`, không phải kết quả |

## Agents (`.claude/agents/`)

Năm agent, theo **pha vận hành** (2026-09-04d): `checker` (verify độc lập checklist §6, gọi trước `orchestrate` và trước `final`; finding → `checker_log.jsonl`, ERROR chặn run) · **`run-monitor`** (agent chính trong lúc run: đọc scheduler/orchestrate/checker log + `nvidia-smi`, báo tiến độ/GPU/lỗi — CHỈ ĐỌC) · `infra` (GPU/env hỏng) · `analyst` (HẬU run: đọc kết quả thật, anomaly/regime, đề xuất) · `researcher` (DORMANT: chỉ khi user yêu cầu đổi methodology/thư viện). Chi tiết: `.claude/AGENT.md`.

## Cấu trúc repo

- `Baseline_LGBM.py` — B0 frozen (306 feature, LightGBM GPU), không sửa.
- `src/p0/` — harness theo plan §8 (`features_short`, `s0`, `gpu`, `scheduler`, `fold_parallel`, `orchestrate`, `lora`, `models_tfm`, `visualize`, `cli`, …); `run.py` CLI; `configs/`; `tests/`; `scripts/`.
- `experiments/15d/` (vòng 15 ngày) · `experiments/full/` (vòng expanded-data: `s0/`, `checker_log.jsonl` đã có; còn lại sinh khi chạy).
- `data/` — **hai CSV canonical 2 năm (Git LFS, có trong repo)** + anchor checksum + sidecar dẫn xuất; CSV khác không track.
- `docs/` — plan chính thức + `reference/` (audit) + `archive/`. `reports/` — smoke visualize (số giả). `.claude/` — hiến pháp, trạng thái, agents (5 vai trò), hooks.
