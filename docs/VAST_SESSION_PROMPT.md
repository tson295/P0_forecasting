# Prompt cho session Claude Code mới trên Vast (copy nguyên khối dưới vào session)

> Dùng sau khi: (1) instance Vast đã tạo, (2) repo đã clone vào `~/P0_forecasting`, (3) hai file CSV
> (`data/BTC_hf_1min.csv`, `data/BTC_lf_5min.csv`) đã scp lên (CSV không nằm trong git).
> **Prompt này CHÍNH LÀ authorization của user để chạy experiment thật**: sau khi mọi preflight PASS, session tự
> chuyển `TRAINING: UNLOCKED` và chạy tiếp, không hỏi lại chỉ để unlock.

---

Bạn là session Claude Code chạy trên máy Vast.ai GPU cho project **P0_forecasting** (BTC 1-phút point forecasting).
Đọc theo thứ tự trước khi làm gì: `.claude/CLAUDE.md` → `.claude/MEMORY.md` → `docs/RESEARCH_PLAN.md` → `README.md`
→ `.claude/AGENT.md` (4 agent: checker, researcher, analyst, infra). `docs/archive/` và `docs/reference/` là tham
khảo/lịch sử, không có hiệu lực; `reports/smoke_visualize.md` là layout mẫu số giả, không phải kết quả.

## Luật bất biến

1. **Training chỉ trên GPU — cấm training CPU, không fallback.** Nhưng KHÔNG hard-code một backend cho mọi thư viện:
   torch/LSTM/TimesFM = CUDA; XGBoost `device=cuda`; CatBoost `task_type=GPU`; LightGBM = `device_type` **thực sự
   fit được** trên máy (`gpu` build OpenCL hoặc `cuda` build CUDA). `scripts/vast_bootstrap.sh` tự resolve và ghi
   backend LightGBM vào `configs/p0_15d.json`; backend đó chảy sang cả AutoTS-WR probe và mọi template bake-off
   (`cli.autots_regressors`). Không có GPU backend hợp lệ cho một model bắt buộc → **DỪNG, hỏi user**.
2. **TRAINING lock**: trước preflight `.claude/MEMORY.md` vẫn `TRAINING: LOCKED`. Chỉ khi ĐỦ các điều kiện ở
   "Preflight" mới tự sửa thành `TRAINING: UNLOCKED` rồi chạy tiếp.
3. Mỗi run phải thuộc một bước của plan §8. Không chạy trùng, không idle GPU (Vast tính giờ). TEST chỉ chạm ở `final`,
   đúng một lần.
4. Không thêm model/metric/feature ngoài plan; không sweep hyperparameter; không sửa `Baseline_LGBM.py`; không đổi
   luật KEEP/DROP (`MedianGain ≥ −ε_m`), R1–R4, champion (`> +ε_champion`), gộp 3 seed, vai trò seed (§1.3:
   `calib_seed` chỉ cho ES; `eval_seeds` đo ε + confirmation; MỘT `selection_seed` cho mọi bước selection).
5. TimesFM giữ: zero-shot (không LoRA), mean head `quantile[...,0]`, covariate `per_core_batch_size=1`,
   1 origin/lời gọi, covariate dịch 1 bar, cộng dồn one-step → `y_1/y_2/y_3`. AutoTS giữ: probe = 2 class cố định,
   framework chỉ chạy với `initial_template` GPU + `max_generations=0`, search chỉ nhìn training-side (FIT+ES).
6. `data/` read-only; không ghi đè `data/data_checksums.json`; không đưa secret vào repo/MEMORY/git.
   Commit + push sau mỗi phase (`git add -A && git commit && git push`); raw CSV bị `.gitignore` loại.

## Preflight (chạy tuần tự, mọi bước phải PASS)

```bash
cd ~/P0_forecasting && tmux new -s p0        # mọi việc dài chạy trong tmux
git log --oneline -1                          # ghi commit hash vào báo cáo
bash scripts/vast_bootstrap.sh                # fail-fast: GPU, apt, pip, timesfm/autots/jax, build+resolve LightGBM,
                                              # preflight XGB/Cat/torch/LGBM, unit test. Exit != 0 = DỪNG.
PYTHONPATH=src:. python scripts/vast_canary.py --config configs/p0_15d.json
python run.py check-data --config configs/p0_15d.json
```

- `vast_canary.py` = canary **package thật** (không stub) trên data tổng hợp: fit GPU thật của LightGBM/XGBoost/
  CatBoost/XGB-RF/LSTM; TimesFM load checkpoint + native forecast + mean-head + covariate batch=1 + causal/shift;
  AutoTS import + WR/MR probe + `future_regressor` thật sự được dùng + bake-off template GPU-safe. Ghi
  `experiments/canary.json` kèm **ETA theo phase**. Bất kỳ FAIL nào → DỪNG, báo user traceback + API mismatch,
  **không tự đổi methodology**.
- `check-data` phải in: HF 21.916 dòng (2026-01-18 16:15 → 02-02 21:30), `ok: true`, B0-eligible 21.258 origin,
  5 fold + TEST `OK` với n: FIT 9.887/11.327/12.767/14.207/15.647 (Final 17.087), ES 1.377, VAL 1.437, TEST 2.728,
  và `verify … OK — khớp snapshot đã ghi`. Sai số bất kỳ hoặc `KHÔNG KHỚP` → DỪNG, hỏi user, **không** chạy
  `--write-checksums` để ghi đè.

Đủ điều kiện unlock: commit đúng · bootstrap PASS · canary PASS · `check-data` PASS · `pytest -q -x` PASS.
→ sửa `.claude/MEMORY.md` thành `TRAINING: UNLOCKED`, commit, rồi chạy tiếp mà không cần hỏi lại.

## Thứ tự chạy (plan §8)

```bash
# Phase A
python run.py calibrate --config configs/p0_15d.json --model lgbm --colset b0306
python run.py filter-b0 --config configs/p0_15d.json

# Phase B — mỗi model: calibrate ES → fixed rounds → add-one 39 → PI prune → confirmation (ES bật lại) → win → champion
python run.py loop --config configs/p0_15d.json --model lgbm     # BẮT BUỘC đầu tiên (champion ban đầu §3)
python run.py loop --config configs/p0_15d.json --model xgb
python run.py loop --config configs/p0_15d.json --model cat
python run.py loop --config configs/p0_15d.json --model tfm_b0   # TimesFM nhánh A: S = B0*
python run.py loop --config configs/p0_15d.json --model tfm_ext  # TimesFM nhánh B: S = ∅ (native r1)
python run.py tfm-final --config configs/p0_15d.json             # TFM_B0_best vs TFM_EXT_best → TimesFM-final
python run.py loop --config configs/p0_15d.json --model xgbrf
python run.py loop --config configs/p0_15d.json --model autots_wr   # probe (không champion/ensemble/Final)
python run.py loop --config configs/p0_15d.json --model autots_mr   # probe
python run.py autots-search --config configs/p0_15d.json            # framework trên F_WR_best và F_MR_best → AutoTS-final
python run.py loop --config configs/p0_15d.json --model lstm

# Phase C
python run.py ensemble --config configs/p0_15d.json
python run.py final --config configs/p0_15d.json                 # TEST một lần duy nhất
```

Output mong đợi sau mỗi bước: `experiments/log.csv` (schema cố định), `calib/<model>_<tag>.json`
(rounds, ε, noise_cells, 3 vai trò seed), `b0_filter.csv` + `b0_sets.csv` + `b0_star.json`,
`keepdrop_<model>.csv` (39 dòng), `prune_pi_<model>.csv`, `prune_<model>.csv`, `wins/<model>.json` +
`<model>_seed{k}.npz` (k = đúng số seed đã chạy), `latency_<model>.csv`, `champion.json`, `champion_log.csv`,
`tfm_final.csv`, `autots_search.csv` + `autots_templates/`, `summary/fig_path_*`, `fig_traj_h{1,2,3}_*`,
`fig_HM_*`, và ở Final: `summary/all_models_test.csv`, `fig_final_heatmaps.png`, `fig_final_paths_all_models.png`,
`fig_final_traj_h{1,2,3}_all_models.png`, `latency_summary.csv`.

## Vận hành

- Chạy liên tục: phase trước PASS thì tự chạy phase sau, **không hỏi user giữa từng model**. Sau mỗi phase: cập nhật
  ETA bằng thời gian thật, commit + push, ghi trạng thái vào `.claude/MEMORY.md` (Current Task / Exact Next Step).
- SSH rớt: `tmux attach -t p0`, đọc `experiments/log.csv` + `wins/` để biết bước nào xong, resume từ bước hợp lệ kế
  tiếp. **Không chạy lại `final`** nếu `summary/all_models_test.csv` đã có.
- Kiểm tra hợp lý khi đọc kết quả: MedianGain vs E0 chỉ cỡ 0.05–0.3 pp; **Gain > ~1 pp → nghi leakage**, dừng và gọi
  agent `checker`. `std(ŷ) ≪ std(y)` và forecast "phẳng" là bình thường. `best_iteration` chạm trần 1200 → ghi chú.
- Agent: `checker` (verify độc lập, có quyền phủ quyết), `researcher` (audit API/version), `analyst` (đọc kết quả
  thật sau full run), `infra` (GPU/env hỏng). Không tạo lại controller/coder/runner.

## Chỉ DỪNG và hỏi user khi

GPU backend không khả dụng cho một model bắt buộc · package/API thật lệch adapter mà không thể sửa nếu không đổi
methodology · checksum/data mismatch · phát hiện leakage hoặc bug correctness mới · test fail mà muốn sửa phải đổi
methodology · OOM/hết dung lượng không xử lý được bằng tinh chỉnh execution an toàn (batch, tail_bars, dọn cache).
Ngoài ra: tự quyết và chạy tiếp.

## Báo cáo (cuối mỗi phase và cuối run)

Bước nào, lệnh gì, thời gian thật vs ETA, GPU, file output, số liệu chính (MedianGain/WinRate/P10/Worst, ε, số vòng,
KEEP/DROP, win, đổi/giữ champion), điều bất thường, việc kế tiếp, commit hash sau khi push.
