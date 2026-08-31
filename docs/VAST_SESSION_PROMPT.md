# Prompt cho session Claude Code mới trên Vast

> Copy nguyên khối dưới đây vào một session Claude Code mới trên máy Vast. Dùng sau khi: instance đã tạo, repo đã
> clone vào `~/P0_forecasting`, và `data/BTC_hf_1min.csv` + `data/BTC_lf_5min.csv` đã scp lên (CSV không nằm trong git).

---

Bạn là session Claude Code chạy trên máy Vast.ai GPU cho project P0_forecasting (BTC 1-phút point forecasting).
Repo đã clone ở ~/P0_forecasting (commit 97c18d4 trở lên), hai file data/BTC_hf_1min.csv và data/BTC_lf_5min.csv
đã được scp lên (CSV không nằm trong git).

PROMPT NÀY CHÍNH LÀ AUTHORIZATION CỦA USER ĐỂ CHẠY TOÀN BỘ EXPERIMENT END-TO-END:
sau khi mọi preflight PASS, bạn TỰ chuyển TRAINING: UNLOCKED và chạy LIÊN TỤC:
Phase A → Phase B → Phase C → Final.
KHÔNG hỏi user để duyệt giữa các phase, giữa các model, sau khi chọn B0*, sau khi xem ETA, hay sau khi báo cáo một phase.
Báo cáo/cập nhật MEMORY/commit/push sau mỗi phase KHÔNG phải approval gate: hoàn tất chúng rồi tự chạy phase kế tiếp.
Chỉ DỪNG và hỏi user trong các blocker được liệt kê rõ ở cuối prompt.

ĐỌC TRƯỚC (theo thứ tự): .claude/CLAUDE.md → .claude/MEMORY.md → docs/RESEARCH_PLAN.md → README.md →
.claude/AGENT.md. docs/archive/ và docs/reference/ chỉ là lịch sử nghiên cứu, KHÔNG có hiệu lực.
reports/smoke_visualize.md là layout mẫu số giả, không phải kết quả.

=========================== LUẬT BẤT BIẾN ===========================
1. Training CHỈ trên GPU, cấm CPU, không fallback âm thầm. NHƯNG không hard-code một backend cho mọi thư viện:
   - torch / LSTM / TimesFM : CUDA
   - XGBoost                : device=cuda
   - CatBoost               : task_type=GPU
   - LightGBM               : device_type = backend THỰC SỰ FIT ĐƯỢC trên máy này ("gpu" build OpenCL HOẶC "cuda"
     build CUDA). scripts/vast_bootstrap.sh tự thử fit và GHI kết quả vào configs/p0_15d.json
     (models.lgbm.device_type). Backend đó phải chảy nhất quán sang AutoTS-WR probe và mọi template LightGBM của
     bake-off — code đã làm việc này qua cli.autots_regressors(cfg); ĐỪNG hard-code lại ở bất cứ đâu.
   Nếu một model bắt buộc không có GPU backend hợp lệ → DỪNG và hỏi user. Cấm: fallback CPU, skip model, thay
   model, đổi methodology.

2. TRAINING lock: trước preflight .claude/MEMORY.md vẫn "TRAINING: LOCKED". CLI tự từ chối training khi còn LOCKED.

3. Mỗi run thuộc đúng một bước của plan §8. Không chạy trùng, không để GPU idle (Vast tính giờ).
   TEST chỉ chạm ở bước `final`, đúng MỘT lần.

4. Không thêm model/metric/feature ngoài plan; không sweep hyperparameter; không sửa Baseline_LGBM.py; không đổi
   luật KEEP/DROP (MedianGain >= -eps_m), R1-R4, champion (> +eps_champion), cách gộp 3 seed, hay vai trò seed §1.3
   (calib_seed CHỈ cho ES lấy số vòng; eval_seeds đo epsilon + confirmation; MỘT selection_seed cho mọi bước
   selection). Không sửa test để né failure.

5. TimesFM: zero-shot, KHÔNG LoRA, KHÔNG dự báo close. Giữ mean head quantile[...,0] (point_forecast là q50),
   covariate compile per_core_batch_size=1, 1 origin mỗi lời gọi, covariate dịch 1 bar, cộng dồn one-step -> y1/y2/y3.
   AutoTS: probe = 2 class cố định; framework chỉ chạy với initial_template GPU do repo khai báo + max_generations=0;
   search chỉ nhìn training-side (FIT+ES, dừng trước purge 60'), outer VAL chỉ để chấm.

6. data/ read-only. KHÔNG ghi đè data/data_checksums.json. Không đưa secret (API key, SSH key, token) vào
   repo/MEMORY/git. Commit + push sau mỗi phase; raw CSV đã bị .gitignore loại.

=========================== PREFLIGHT (fail-fast) ===========================
Chạy tuần tự trong tmux; bước nào exit != 0 thì DỪNG và báo user kèm log:

  cd ~/P0_forecasting && tmux new -s p0
  git log --oneline -1                       # ghi commit hash vào báo cáo
  bash scripts/vast_bootstrap.sh             # GPU, apt OpenCL/boost, pip, cài timesfm[torch]==2.0.2 +
                                             # autots==1.0.4 + statsmodels + jax[cpu], BUILD + RESOLVE backend
                                             # LightGBM rồi ghi vào config, preflight torch/XGB/Cat/LGBM, unit test
  PYTHONPATH=src:. python scripts/vast_canary.py --config configs/p0_15d.json
  python run.py check-data --config configs/p0_15d.json
  PYTHONPATH=src:. python -m pytest -q -x

vast_canary.py là canary PACKAGE THẬT (không stub), chạy trên data tổng hợp nên không đụng data thật và không phải
training của experiment. Nó kiểm: fit GPU thật của LightGBM/XGBoost/CatBoost/XGB-RF/LSTM; TimesFM import + load
đúng checkpoint/revision + native forecast + mean-head = quantile[...,0] + covariate với per_core_batch_size=1 +
1 origin/call + cắt-chuỗi-tại-t cho prediction bit-identical + covariate dịch đúng 1 bar + output hữu hạn; AutoTS
import + WR probe (LightGBM GPU) + MR probe (XGBoost cuda) + future_regressor THẬT SỰ được dùng + bake-off
top-level với mọi template GPU-safe. Nó ghi experiments/canary.json kèm ETA theo phase.

Nếu package/API thật lệch adapter -> DỪNG, báo user exact traceback + chỗ mismatch. TUYỆT ĐỐI không tự đổi
methodology, không sửa adapter cho "chạy được" nếu điều đó đổi ngữ nghĩa.

check-data phải in đúng: HF 21.916 dòng (2026-01-18 16:15 -> 02-02 21:30), ok: true, B0-eligible 21.258 origin,
5 fold + TEST đều OK với n: FIT 9.887 / 11.327 / 12.767 / 14.207 / 15.647 (Final 17.087), ES 1.377, VAL 1.437,
TEST 2.728, và dòng cuối "verify ... OK — khớp snapshot đã ghi". Sai bất kỳ số nào, hoặc KHÔNG KHỚP, hoặc thiếu
data_checksums.json -> DỪNG hỏi user. KHÔNG chạy --write-checksums để ép PASS.

TỰ UNLOCK khi và chỉ khi đủ 5 điều: commit đúng · bootstrap PASS · canary PASS · check-data PASS · pytest PASS.
Khi đó: sửa .claude/MEMORY.md thành "TRAINING: UNLOCKED", commit, rồi chạy tiếp mà không hỏi lại.

=========================== THỨ TỰ CHẠY (plan §8) ===========================
Phase A:
  python run.py calibrate --config configs/p0_15d.json --model lgbm --colset b0306
  python run.py filter-b0 --config configs/p0_15d.json

Sau khi Phase A PASS:
- ghi kết quả/ETA;
- cập nhật MEMORY;
- commit + push;
- NGAY LẬP TỨC tiếp tục Phase B.
KHÔNG chờ user duyệt B0*, feature count, kết quả Phase A hay ETA.

Phase B (mỗi model: calibrate ES -> fixed rounds/epoch -> sequential add-one 39 candidate, mỗi candidate FIT LẠI
trên S+f -> PI prune -> confirmation refit F* và F*_prune với ES bật lại trên eval_seeds -> win_m -> champion):

  python run.py loop --config configs/p0_15d.json --model lgbm       # BẮT BUỘC đầu tiên (champion ban đầu §3)
  python run.py loop --config configs/p0_15d.json --model xgb
  python run.py loop --config configs/p0_15d.json --model cat
  python run.py loop --config configs/p0_15d.json --model tfm_b0     # TimesFM nhánh A: S = B0*
  python run.py loop --config configs/p0_15d.json --model tfm_ext    # TimesFM nhánh B: S = rỗng (native r1)
  python run.py tfm-final --config configs/p0_15d.json               # TFM_B0_best vs TFM_EXT_best -> TimesFM-final
  python run.py loop --config configs/p0_15d.json --model xgbrf
  python run.py loop --config configs/p0_15d.json --model autots_wr  # probe: KHÔNG champion/ensemble/Final
  python run.py loop --config configs/p0_15d.json --model autots_mr  # probe
  python run.py autots-search --config configs/p0_15d.json           # framework(F_WR_best) vs framework(F_MR_best)
                                                                     # dedup nếu trùng; chọn ở selection_seed;
                                                                     # freeze winner; confirm trên eval_seeds;
                                                                     # eps_AutoTS của chính winner -> AutoTS-final
  python run.py loop --config configs/p0_15d.json --model lstm

Sau khi Phase B PASS:
- ghi kết quả/ETA;
- cập nhật MEMORY;
- commit + push;
- NGAY LẬP TỨC tiếp tục Phase C.
KHÔNG chờ user duyệt model/champion/feature set/kết quả trung gian.

Phase C:
  python run.py ensemble --config configs/p0_15d.json
  python run.py final --config configs/p0_15d.json                   # TEST đúng một lần

=========================== VẬN HÀNH ===========================
- GOAL = hoàn thành TOÀN BỘ pipeline đến hết `final`, không phải chỉ hoàn thành phase hiện tại.

- Chạy liên tục trong tmux:
  preflight PASS → tự chạy Phase A → tự chạy Phase B → tự chạy Phase C → Final.
  KHÔNG hỏi user giữa phase, giữa model hoặc trước Final nếu không có blocker thuộc danh sách cuối.

- Sau mỗi phase: cập nhật ETA bằng thời gian THẬT (so với experiments/canary.json), commit + push, ghi trạng thái
  vào .claude/MEMORY.md (Current Task / Exact Next Step), rồi TỰ ĐỘNG chạy phase kế tiếp.
  Báo cáo phase chỉ là progress report, KHÔNG phải yêu cầu user approve.
  Không fake độ chính xác của ETA.

- Trước full run, in bảng ETA theo phase: Phase A, LGBM, XGB, Cat, tfm_b0, tfm_ext, XGB-RF, AutoTS-WR probe,
  AutoTS-MR probe, autots-search, LSTM, ensemble/final, tổng.

- ETA, kể cả ETA lớn của `tfm_b0`, CHỈ DÙNG ĐỂ BÁO CÁO.
  Không dừng và không hỏi user chỉ vì một phase/model chạy lâu hoặc tốn nhiều giờ hơn dự kiến.
  `tfm_b0` là một phần bắt buộc của experiment và phải chạy như plan nếu không gặp blocker kỹ thuật/correctness
  nằm trong danh sách "CHỈ DỪNG VÀ HỎI USER KHI" bên dưới.
  Không tự bỏ nhánh, không đổi feature set, không đổi methodology để giảm runtime.

- SSH rớt: tmux attach -t p0; đọc experiments/log.csv + experiments/wins/ để biết bước nào đã xong; resume từ bước
  hợp lệ kế tiếp. KHÔNG chạy lại `final` nếu summary/all_models_test.csv đã tồn tại.

- Đọc kết quả có kiểm chứng: MedianGain vs E0 chỉ cỡ 0.05–0.3 pp là bình thường; Gain > ~1 pp vs B0/E0 -> NGHI
  LEAKAGE, dừng và gọi agent checker. std(yhat) << std(y) và forecast trông "phẳng" là bình thường.
  best_iteration chạm trần 1200 -> ghi chú, không đổi config.

- Agent chỉ có 4: checker (verify độc lập, có quyền phủ quyết), researcher (audit API/version), analyst (đọc kết
  quả thật sau full run), infra (GPU/env hỏng). KHÔNG tạo lại main-controller/coder/runner; pipeline deterministic
  do chính session này chạy.

=========================== CHỈ DỪNG VÀ HỎI USER KHI ===========================
- không có GPU backend khả dụng cho một model bắt buộc;
- package/API thật lệch adapter mà không thể sửa nếu không đổi methodology;
- checksum/data mismatch;
- phát hiện leakage hoặc bug correctness mới;
- test fail mà muốn sửa thì phải đổi methodology;
- OOM/hết dung lượng không xử lý được bằng tinh chỉnh execution an toàn (batch size, tail_bars, dọn cache HF).

KHÔNG phải lý do để dừng:
- Phase A/B/C vừa hoàn thành;
- B0* đã được chọn;
- một model vừa hoàn thành;
- champion vừa thay đổi/không thay đổi;
- ETA cao hoặc runtime lâu;
- tfm_b0 có nhiều covariate;
- cần báo cáo progress;
- cần commit/push sau phase.

Ngoài các blocker kỹ thuật/correctness được liệt kê bên trên: TỰ QUYẾT và CHẠY TIẾP cho tới khi `final` hoàn tất.

=========================== BÁO CÁO ===========================
Cuối mỗi phase và cuối run, báo: bước nào + lệnh gì, thời gian thật vs ETA, GPU đang dùng, file output sinh ra,
số liệu chính (MedianGain/WinRate/P10Gain/WorstGain, epsilon, số vòng/epoch, KEEP/DROP, win, đổi/giữ champion),
điều bất thường, việc kế tiếp, và commit hash sau khi push.

Báo cáo giữa các phase KHÔNG yêu cầu user phản hồi. Sau khi ghi báo cáo, tự chạy bước tiếp theo ngay.

Bắt đầu bằng: đọc 5 file context ở trên, `git log --oneline -1`, rồi chạy preflight.
