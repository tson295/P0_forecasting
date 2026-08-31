# Prompt cho session Claude Code mới trên Vast (copy nguyên khối dưới vào session)

> Dùng sau khi: (1) instance Vast đã tạo, (2) repo đã clone vào `~/P0_forecasting`, (3) file `data/BTC_hf_1min.csv` và `data/BTC_lf_5min.csv`
> đã scp lên (CSV không nằm trong git). User vẫn phải **unlock training** bằng lệnh rõ trong session mới; prompt này không thay lệnh đó.

---

Bạn là session Claude Code chạy trên máy Vast.ai GPU cho project **P0_forecasting** (BTC 1-phút point forecasting). Đọc theo thứ tự trước khi làm gì:
`.claude/CLAUDE.md` (hiến pháp rút gọn) → `.claude/MEMORY.md` (trạng thái, `TRAINING:`) → `docs/RESEARCH_PLAN.md` (plan chính thức, rev 9b) → `README.md` → `.claude/AGENT.md` (agent: main-controller, coder, researcher, checker, runner, analyst, infra). `docs/archive/` và `docs/reference/` không có hiệu lực. `reports/smoke_visualize.md` là layout mẫu với số giả, không phải kết quả.

## Luật bất biến

1. **Training chỉ trên GPU — cấm training CPU, không fallback.** LightGBM phải là build GPU (`device_type=gpu`, hoặc `cuda` nếu OpenCL không có); XGBoost `device=cuda`; CatBoost `task_type=GPU`; torch CUDA. CPU chỉ cho tính feature, metric, MI/PI, unit test, và predict của thư viện mặc định chạy CPU. `--smoke`/`--allow-cpu` chỉ có tác dụng với dataset tổng hợp — CLI từ chối trên data thật; không tìm cách lách.
2. **TRAINING lock**: chỉ chạy `calibrate / filter-b0 / loop / ensemble / final` khi `.claude/MEMORY.md` ghi `TRAINING: UNLOCKED` do user ra lệnh rõ ("unlock training" / "bắt đầu training" / "run experiments"). CLI tự kiểm tra và từ chối nếu LOCKED. Chưa unlock → chỉ bootstrap, check-data, unit test.
3. Mỗi run phải thuộc một bước của plan §8 và trả lời "thuộc bước nào, so với base nào, dùng số vòng/ε của model nào". Không chạy trùng, không idle GPU (Vast tính giờ), không rerun vì quên config (config/log đã persist trong `experiments/`).
4. TimesFM/AutoTS: giữ đúng ràng buộc đã ghi trong adapter — TFM covariate **1 origin/lời gọi** + covariate dịch 1 bar; AutoTS regressor dịch theo model (MR `f(s−1)`, WR `f(s+window−1)`), không dùng `AutoTS(...)` (search), không sửa site-packages. Không thêm model, metric, feature ngoài plan; không sweep hyperparameter; không sửa `Baseline_LGBM.py`; không đổi luật KEEP/DROP (`MedianGain ≥ −ε_m` KEEP, `< −ε_m` DROP), R1–R4, champion (`> +ε_champion`), gộp 3 seed (mean RMSE từng ô → Gain → median 15 ô). Không đổi vai trò seed (§1.3): `calib_seed` chỉ cho ES lấy số vòng; `eval_seeds` (3) chỉ để đo ε và confirmation; **mọi bước selection dùng đúng một `selection_seed`** — không được đổi seed giữa các Rk/candidate.
5. `data/` read-only; không đưa secret (Vast API key, SSH key) vào repo/MEMORY; IP/instance id không thành memory. Commit + push sau mỗi bước hoàn tất (`git add -A && git commit && git push`), raw CSV bị `.gitignore` loại.
6. Sau mỗi bước: cập nhật `.claude/MEMORY.md` (Current Task / Exact Next Step / Experiment Findings chỉ khi có run thật) — MEMORY là trạng thái, không phải log.

## Bước 0 — Bootstrap và kiểm tra (chưa cần unlock)

```bash
cd ~/P0_forecasting
tmux new -s p0            # mọi việc dài chạy trong tmux, sống sót SSH disconnect
bash scripts/vast_bootstrap.sh          # apt OpenCL/boost, pip, LightGBM build GPU, preflight GPU, unit test
cat experiments/env.txt                 # ghi GPU, version thư viện, kết quả preflight → dán vào MEMORY (Data/Implementation Blockers)
python run.py check-data --config configs/p0_15d.json     # §1.1 + verify sha256 với data/data_checksums.json đã commit
```
`check-data` phải in: HF 21.916 dòng (2026-01-18 16:15 → 02-02 21:30), `ok: true`, B0-eligible 21.258 origin (01-19 02:46 → 02-02 21:27), 5 fold + TEST đều `OK` với n: FIT 9.887 / 11.327 / 12.767 / 14.207 / 15.647 (Final 17.087), ES 1.377, VAL 1.437, TEST 2.728; dòng cuối `verify … OK — khớp snapshot đã ghi`. Sai bất kỳ số nào, hoặc `KHÔNG KHỚP` / `Thiếu data_checksums.json` → file scp lên không phải snapshot đã kiểm → DỪNG, báo user, không training, KHÔNG chạy `--write-checksums` để ghi đè.

Nếu preflight LightGBM `device_type=gpu` fail nhưng `cuda` OK → đặt `"lgbm": {"device_type": "cuda"}` trong `configs/p0_15d.json` (vẫn GPU). Cả hai fail → DỪNG.

## Bước 1–4 — chỉ sau khi user unlock (main session sửa MEMORY thành `TRAINING: UNLOCKED`)

Chạy tuần tự, mỗi lệnh trong tmux, log ra `experiments/`:

```bash
# Phase A: calibrate LightGBM trên B0-306 (15fixed_306 + ε) → lọc B0 → B0*
python run.py calibrate --config configs/p0_15d.json --model lgbm --colset b0306
python run.py filter-b0 --config configs/p0_15d.json          # PI + SA (306 model 1 cột, ~2–4 h) + MI → R1–R4 → experiments/b0_star.json

# Phase B/C: từng model từ B0* (calibrate riêng → 39 candidate → prune PI → 3 seed → win_m → latency → champion + figure)
python run.py loop --config configs/p0_15d.json --model lgbm   # bắt buộc đầu tiên: champion ban đầu = LightGBM (CLI từ chối model khác khi chưa có champion)
python run.py loop --config configs/p0_15d.json --model xgb
python run.py loop --config configs/p0_15d.json --model cat
# TimesFM + AutoTS: adapter đã code theo docs/reference/audit_timesfm.md / audit_autots.md, nhưng PACKAGE CHƯA CÀI.
# Cài chỉ khi user cho phép (plan §2.2), rồi smoke import trước khi chạy loop:
#   pip install "timesfm[torch]==2.0.2"                 # + "jax[cpu]" scikit-learn nếu chạy covariate
#   pip install autots==1.0.4 statsmodels               # CHƯA xác minh với pandas 3.0.3 → smoke import trước
python run.py loop --config configs/p0_15d.json --model tfm
python run.py loop --config configs/p0_15d.json --model xgbrf
python run.py loop --config configs/p0_15d.json --model autots_wr
python run.py loop --config configs/p0_15d.json --model autots_mr
python run.py loop --config configs/p0_15d.json --model lstm

python run.py ensemble --config configs/p0_15d.json           # §3 ensemble vs champion
python run.py final --config configs/p0_15d.json              # §4 TEST một lần: all_models_test.csv + heatmap + Fig H_h mọi model
```

Output cần có sau mỗi bước (kiểm tra bằng checklist §6 plan trước khi sang bước sau; dùng agent `checker` nếu nghi ngờ):
- `experiments/calib/<model>_<tag>.json` (rounds 15 ô, ε, rmse, e0), `experiments/log.csv` (mỗi run một dòng, schema cố định: RMSE/MAE/E0/Gain 15 ô, config_hash, train_device), `experiments/runs/<exp_id>/run.json` (+ `pred_val.npz` ở confirmation, `pred_test.npz` ở final).
- `experiments/b0_filter.csv` (306 dòng: PI/SA/MI per h, cờ ≥ 2/3, keep_R1..R4), `b0_sets.csv` (4 run kiểm chứng + bộ được chọn), `b0_star.json`.
- `experiments/keepdrop_<model>.csv` (39 dòng), `prune_pi_<model>.csv`, `prune_<model>.csv` (unprune vs prune, RMSE̅ 3 seed, MedianGain, win), `wins/<model>.json` + `<model>_seed{0,1,2}.npz`, `latency_<model>.csv` (p95/p99/max, train/predict device), `champion.json`, `champion_log.csv` (schema cố định: win_m, RMSE̅ hai bên, Gain 15 ô, metric per horizon, latency), `summary/latency_summary.csv` (VAL/TEST, device, lib version), `summary/fig_path_<model>_vs_champion.png` (forecast path 3 origin), `summary/fig_traj_h{1,2,3}_<model>_vs_champion.png` (trajectory toàn bộ VAL), `summary/fig_HM_<model>_vs_champion.png`.
- Final: `summary/all_models_test.csv` (RMSE/MAE/r/dir-acc, Gain vs E0 / B0-306 / B0* / champion, latency p95/p99/max per h), `summary/fig_final_heatmaps.png`, `summary/fig_final_paths_all_models.png`, `summary/fig_final_traj_h{1,2,3}_all_models.png`, `summary/latency_final_*.csv`.

## Kiểm tra hợp lý (dừng và báo user nếu vi phạm)

- `MedianGain vs E0` của B0-306 và mọi model chỉ cỡ 0.05–0.3 pp; **Gain > ~1 pp vs B0/E0 → nghi leakage/bug**, không tin, chạy `checker`.
- `std(ŷ) ≪ std(y)` là bình thường (tín hiệu 1 phút rất nhỏ). Prediction trông "phẳng" không phải lỗi.
- `best_iteration` chạm trần 1200 → ghi chú, không đổi config.
- Fold/TEST không được đọc chéo: TEST chỉ ở `final`, chạy đúng một lần, không sửa gì sau khi xem.
- Latency chỉ để theo dõi; không đưa vào bất kỳ quyết định nào.

## Báo cáo về cho user (cuối mỗi bước)

Bước nào, lệnh gì, thời gian chạy, GPU, file output, số liệu chính (MedianGain/WinRate/P10/Worst, ε, số vòng, quyết định KEEP/DROP/win/champion), điều bất thường, việc kế tiếp theo §8. Commit hash sau khi push.
