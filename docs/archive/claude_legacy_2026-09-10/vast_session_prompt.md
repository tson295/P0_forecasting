# Prompt cho session Claude Code trên Vast — vòng EXPANDED-DATA (2026-09-03; cập nhật 2026-09-04d: 2 GPU, data trong repo, pha vận hành)

> Chỉ dùng khi user đã **cho phép rõ ràng** chạy experiment vòng expanded-data (MEMORY hiện ghi `TRAINING: LOCKED`).
> Copy nguyên khối dưới đây vào một session Claude Code mới trên máy Vast sau khi: instance đã tạo và repo đã clone vào
> `~/P0_forecasting` bằng `git clone … && cd P0_forecasting && git lfs install && git lfs pull`.
> **Data 2 năm nằm TRONG repo qua Git LFS** — `data/BTC_1m_2y.csv` (sha256 559ce040…f097, 1.051.201 dòng) và
> `data/BTC_5m_2y.csv` (sha256 0e5fb9ad…f2fef, 210.239 dòng) có sẵn sau `git lfs pull`. KHÔNG scp, KHÔNG cần `derive-lf`
> (lệnh đó ở lại làm công cụ kiểm chứng). Nếu quên `git lfs pull`, hai file chỉ là pointer ~130 byte → `check-data` báo checksum lệch.

---

Bạn là session Claude Code chạy trên máy Vast.ai GPU cho project P0_forecasting (BTC 1-phút point forecasting), vòng
EXPANDED-DATA trên DATA 2 NĂM. Repo ở ~/P0_forecasting; CẢ HAI file data (1m + 5m) đã có sẵn trong repo qua Git LFS.

PROMPT NÀY LÀ AUTHORIZATION CỦA USER ĐỂ CHẠY TOÀN BỘ EXPERIMENT END-TO-END (user đã unlock bằng cách gửi prompt này):
sau khi mọi preflight PASS, bạn TỰ chuyển TRAINING: UNLOCKED và chạy LIÊN TỤC theo thứ tự plan §8 tới hết `final` + `visualize`.
KHÔNG hỏi user để duyệt giữa các bước/model. Chỉ DỪNG và hỏi user trong các blocker liệt kê ở cuối prompt.

ĐỌC TRƯỚC (theo thứ tự): .claude/CLAUDE.md → .claude/MEMORY.md → docs/RESEARCH_PLAN.md (rev 10.4) → README.md → .claude/AGENT.md
→ docs/reference/audit_timesfm_lora.md. docs/archive/ là lịch sử; experiments/15d/ là vòng 15 ngày đã xong (không sửa).

=========================== LUẬT BẤT BIẾN (không tương tác) ===========================
0. Bất biến cứng do code tự ép (checker_log.hard_fail → ERROR + dừng): checksum lệch, LF không phủ HF, biên/purge sai, artifact
   S0/Candidate_m malformed hoặc audit trên dataset khác, GPU không có / backend không CUDA, TEST lần hai, TRAINING LOCKED. Khi gặp:
   KHÔNG hỏi user "tiếp hay dừng" — sửa nếu là lỗi code/env rồi chạy lại, hoặc dừng và báo cáo nếu thuộc danh sách blocker cuối prompt.
   Finding tư vấn (WARN/INFO trong experiments/full/checker_log.jsonl: tương quan cao, gain bất thường, runtime) → ghi, tiếp tục.
1. Training CHỈ trên GPU, cấm CPU, không fallback âm thầm; backend từng thư viện do scripts/vast_bootstrap.sh RESOLVE bằng fit thật
   và ghi vào configs/p0_full.json (LightGBM device_type gpu|cuda → chảy sang AutoTS-WR và template bake-off). xreg của TimesFM
   chạy jax GPU (jax[cuda12]==0.11.1, XLA_PYTHON_CLIENT_PREALLOCATE=false). Thiếu GPU backend cho một model bắt buộc → DỪNG hỏi user.
2. TRAINING lock: CLI tự từ chối khi MEMORY còn LOCKED. TEST chỉ chạm ở `final`, đúng MỘT lần.
3. Mỗi run thuộc đúng một bước §8; không chạy trùng, không để GPU idle.
4. Không thêm model/metric/feature ngoài plan; không sweep hyperparameter; không sửa Baseline_LGBM.py; không đổi luật KEEP/DROP,
   prune (chỉ cột mới), confirmation, champion, ensemble, vai trò seed §1.3, S0_m khoá, C_short. Không sửa test để né failure.
5. TimesFM: calibrate = LoRA FIT + ES chọn epoch per fold (VAL không thấy) → freeze → XReg search trên CÙNG adapter → F_raw → prune PI →
   F_pruned → confirmation raw vs pruned → F_win → RỒI MỚI dựng hệ thống A = TimesFM-LoRA baseline (0 feature, 0 B0*, 0 covariate) →
   tfm-final so HAI HỆ THỐNG HOÀN CHỈNH: B = {LoRA + XReg(F_win)} vs A = {LoRA baseline}. KHÔNG gọi là "XReg vs LoRA".
   loss MSE trên ŷ_h; không torch.compile; inject sau load_checkpoint; giữ mean head, 1 origin/lời gọi, dịch 1 bar, cộng dồn one-step.
   Artifact: wins/tfm_lora_baseline.json, wins/tfm_lora_xreg.json → tfm-final → wins/tfm.json (chỉ TFM-final vào champion).
   AutoTS: probe = 2 class cố định từ S0 của nhánh; framework chỉ với initial_template GPU + max_generations=0 trên FIT+ES.
6. data/ read-only: KHÔNG sửa data/BTC_1m_2y.csv hay data/BTC_5m_2y.csv (đã commit qua Git LFS), KHÔNG ghi đè
   data/data_checksums_2y.json và data/BTC_5m_2y.derivation.json. Không secret vào repo/MEMORY/git.
   experiments/** KHÔNG được ignore: commit + push (LFS cho .npz/.pt) sau mỗi model; adapter LoRA trong experiments/full/lora/.
7. KHÔNG vẽ figure trong bất kỳ bước training nào; figure chỉ sinh bằng `python run.py visualize` sau `final`.
8. `final` chỉ chạy MỘT lần (final/TEST_SENTINEL.json). KHÔNG BAO GIỜ dùng --force-test-rerun trừ khi user ra lệnh rõ (recovery).
9. LỊCH CHẠY/GPU CHỈ ĐỔI WALL-CLOCK (2026-09-04c). Máy có 2 × RTX 5000 Ada 32 GB = HAI GPU ĐỘC LẬP (không gộp 64 GB), worker ĐỐI XỨNG:
   KHÔNG gán GPU0 = ML / GPU1 = DL, không pin model family. Mặc định 1 task nặng/GPU (gpu_slots_per_device = 1) — không tự tăng.
   Nhiều nhánh model độc lập được chạy song song (`orchestrate`), nhưng candidate trong một model VẪN TUẦN TỰ và champion được HOÃN
   tới `champion-replay` (thứ tự cố định lgbm→xgb→cat→tfm→xgbrf→autots→lstm). Không đổi seed/hyperparameter/batch/metric để chạy nhanh hơn.
   OOM → giảm gpu_slots_per_device hoặc chạy nhánh nặng riêng; TUYỆT ĐỐI không CPU fallback, không đổi batch LoRA nếu không bắt buộc (ghi rõ khi đổi).
10. SỰ CỐ TÀI NGUYÊN GPU = TÌNH HUỐNG DUY NHẤT ĐƯỢC DỪNG VÀ HỎI USER (quyết định user 2026-09-04d).
   GPU không có / GPU được giao biến mất / CUDA hỏng / backend không train được trên GPU / phát hiện CPU fallback /
   định tuyến GPU sai (UUID trùng) / worker CUDA chết / OOM chặn đường GPU → code gọi `checker_log.gpu_stop`:
   dừng an toàn, giữ artifact đã xong, KHÔNG CPU fallback, KHÔNG tự đổi batch/hyperparameter/seed/methodology,
   ghi ERROR `ref=USER_DECISION_REQUIRED`, exit code 3. Khi thấy exit 3 hoặc ERROR đó: BÁO CÁO NGUYÊN VĂN CHO USER
   kèm phương án (sửa/đổi GPU rồi chạy lại đúng bước, chạy 1 GPU với P0_GPU_DEVICES=0, hoặc dừng) và CHỜ user chọn.
   Không tự chọn, không tự chạy lại bằng CPU, không tự giảm batch. Mọi vi phạm bất biến KHOA HỌC khác (checksum,
   leakage, biên, S0 malformed, TEST lần hai) vẫn dừng TỰ ĐỘNG, KHÔNG hỏi, không có tuỳ chọn "chạy tiếp".
11. AGENT theo pha VẬN HÀNH: `checker` trước `orchestrate` và trước `final`; `run-monitor` theo dõi trong lúc chạy
   (chỉ đọc scheduler_log/orchestrate_log/checker_log + nvidia-smi); `infra` khi GPU/env hỏng; `analyst` SAU khi có
   artifact thật; `researcher` KHÔNG gọi trong đường chạy này.

=========================== PREFLIGHT (fail-fast) ===========================
  cd ~/P0_forecasting && git lfs install && git lfs pull && tmux new -s p0
  git log --oneline -1
  git lfs ls-files | grep BTC_          # phải thấy CẢ data/BTC_1m_2y.csv lẫn data/BTC_5m_2y.csv
  ls -l data/BTC_1m_2y.csv data/BTC_5m_2y.csv    # ~101,8 MB và ~21,1 MB (nếu ~130 byte = quên `git lfs pull`)
  export P0_GPU_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false     # 2 GPU đối xứng, 1 task nặng/GPU (KHÔNG đặt P0_FOLD_WORKERS)
  nvidia-smi --query-gpu=index,name,memory.total --format=csv        # phải thấy đúng 2 × RTX 5000 Ada 32 GB
  bash scripts/vast_bootstrap.sh                                    # GPU, pip, timesfm 2.0.2 + autots 1.0.4 + jax[cuda12], build/resolve LightGBM, preflight, unit test
  PYTHONPATH=src:. python scripts/vast_canary.py --config configs/p0_full.json
  PYTHONPATH=src:. python scripts/canary_xreg_gpu.py --config configs/p0_full.json
  python run.py gpu-probe --config configs/p0_full.json             # worker 0 → GPU vật lý 0, worker 1 → GPU vật lý 1; UUID PHẢI KHÁC NHAU;
                                                                   # probe backend THẬT trong từng worker (torch/xgboost/lightgbm/catboost/jax/timesfm);
                                                                   # lỗi GPU ở đây = dừng exit 3 → hỏi user (KHÔNG tự sửa bằng CPU)
  # (tuỳ chọn kiểm chứng) python run.py derive-lf --config configs/p0_full.json --force  → phải tái lập ĐÚNG sha 0e5fb9ad…f2fef
  python run.py check-data --config configs/p0_full.json            # KHÔNG --write-checksums: anchor data/data_checksums_2y.json đã commit → phải in "verify … OK"
  python run.py lock-s0 --config configs/p0_full.json               # S0_m khoá + overlap audit + Candidate_m → experiments/full/s0/ (audit label = dataset 2 năm)
  PYTHONPATH=src:. python -m pytest -q -x

check-data phải in: HF 1.051.201 bar 2024-09-03 16:29 → 2026-09-03 16:29 ok:true; LF 210.239 bar dẫn xuất từ đúng HF; B0-eligible 1.049.358;
split rolling_spread 5 fold: VAL 2025-01-07, 2025-05-30, 2025-10-19, 2026-03-11, 2026-08-01 (mỗi 3 ngày, FIT 120 ngày ≈ 172,7k origin, ES ≈ 7,1k,
VAL ≈ 4,3k), final_TEST 2026-08-04 16:30 → 09-03 16:30 (42.918 origin), tất cả OK, và "verify … OK — khớp snapshot đã ghi".
lock-s0 phải in: S0 (locked_b0 + locked_ext): lgbm 72+14, xgb 72+11, cat 72+5, xgbrf 72+12, lstm 72+23, autots_wr 72+21, autots_mr 72+8, tfm 0+0;
C_short 163; Candidate_m = 163 cho mọi model (0 overlap); near vs S0 chỉ báo (không bỏ). Sau đó: `python scripts/checker_record.py --exp experiments/full --blocking`
phải sạch ERROR. Sai bất kỳ số nào / checksum KHÔNG KHỚP → DỪNG hỏi user; KHÔNG chạy --write-checksums để ép PASS.
Canary LoRA thời gian thật (SAU khi unlock, TRƯỚC khi chạy nhánh model): `PYTHONPATH=src:. python scripts/canary_lora.py --config configs/p0_full.json`
— 1 fold × 1 epoch × 64 origin, adapter vào thư mục tạm, KHÔNG ghi artifact/log của experiment; ghi thời gian/epoch, VRAM đỉnh,
ms/origin XReg và ETA vào báo cáo. Nếu OOM khi 2 task nặng cùng chạy → đặt P0_GPU_DEVICES=0 cho bước tfm (1 task nặng), KHÔNG đổi batch LoRA
nếu không bắt buộc (ghi rõ khi đổi), KHÔNG bao giờ chuyển sang CPU.

TỰ UNLOCK khi và chỉ khi đủ: commit đúng · bootstrap PASS · hai canary PASS · check-data PASS · lock-s0 PASS · pytest PASS · checker_log sạch ERROR.
Khi đó: sửa .claude/MEMORY.md thành "TRAINING: UNLOCKED", commit, rồi chạy tiếp mà không hỏi lại.

=========================== THỨ TỰ CHẠY (plan §8) ===========================
  # KHUYẾN NGHỊ trên máy 2 GPU — một lệnh, DAG nhánh, hai GPU luôn có việc, champion replay ở cuối:
  python run.py orchestrate --config configs/p0_full.json --dry-run    # in DAG rồi thoát (kiểm tra trước)
  python run.py orchestrate --config configs/p0_full.json              # loop lgbm/xgb/cat/tfm/xgbrf/autots_wr/autots_mr/lstm (song song theo GPU rảnh)
                                                                       # → tfm-final (ngay khi nhánh tfm xong) + autots-search (ngay khi CẢ hai probe xong)
                                                                       # → champion-replay (thứ tự cố định, chỉ đọc artifact) → ensemble. KHÔNG chạm TEST.
  # HOẶC từng bước (tương đương; dùng khi cần theo dõi/thử từng model):
  python run.py loop --config configs/p0_full.json --model lgbm        # champion ban đầu §3 (khi replay: lgbm vẫn là mốc đầu)
  python run.py loop --config configs/p0_full.json --model xgb
  python run.py loop --config configs/p0_full.json --model cat
  python run.py loop --config configs/p0_full.json --model tfm         # calibrate = LoRA FIT+ES → freeze → XReg add-one → prune → confirmation → F_win + hệ thống A
  python run.py tfm-final --config configs/p0_full.json                # B {LoRA + XReg(F_win)} vs A {LoRA baseline} → TimesFM-final
  python run.py loop --config configs/p0_full.json --model xgbrf
  python run.py loop --config configs/p0_full.json --model autots_wr   # probe (S0 = bộ thắng WR cũ)
  python run.py loop --config configs/p0_full.json --model autots_mr   # probe (S0 = bộ thắng MR cũ)
  python run.py autots-search --config configs/p0_full.json            # framework(F_WR_best) vs framework(F_MR_best) → AutoTS-final
  python run.py loop --config configs/p0_full.json --model lstm
  python run.py champion-replay --config configs/p0_full.json          # champion theo THỨ TỰ CỐ ĐỊNH, chỉ đọc artifact (defer_champion: true)
  python run.py ensemble --config configs/p0_full.json
  python run.py final --config configs/p0_full.json                    # TEST đúng một lần; tuần tự, không qua scheduler; lưu final/*.npz
  python run.py visualize --config configs/p0_full.json                # hậu kỳ: mọi figure từ artifact

Sau mỗi model: cập nhật MEMORY (Current Task / Exact Next Step, thời gian thật vs ETA), `git add -A && git commit && git push`
(LFS cho .npz/.pt), rồi chạy tiếp ngay. `loop --resume` nếu SSH rớt giữa add-one (đọc log.csv + calib/<m>_base.json).

=========================== ĐỌC KẾT QUẢ ===========================
- MedianGain vs E0 chỉ cỡ 0.05–0.3 pp là bình thường; Gain > ~1 pp vs B0/E0 → NGHI LEAKAGE, dừng và gọi agent checker.
- TimesFM-LoRA ≈ E0 là kết quả hợp lệ (r1 gần nhiễu trắng); ES trên ES-partition là chốt chặn.
- checker_log WARN (UNUSUAL_GAIN, C_SHORT_INTRA_IDENTICAL, near) chỉ ghi nhận; ERROR (hard invariant) = run đã tự dừng → sửa rồi chạy lại.
- experiments/full/scheduler_log.jsonl: xem GPU nào bận/rảnh (duration_sec, gpu_physical_id, queue_wait_sec). Nếu một GPU idle nhiều
  trong khi GPU kia đầy việc → BÁO CÁO trong report (để chỉnh scheduling sau), KHÔNG tự đổi methodology/hyperparameter để "cân" máy.
- KHÔNG chạy lại `final` nếu experiments/full/summary/all_models_test.csv đã tồn tại.

=========================== CHỈ DỪNG VÀ HỎI USER KHI ===========================
- **SỰ CỐ TÀI NGUYÊN GPU (ngoại lệ chính thức, luật 10)**: exit code 3 / ERROR `ref=USER_DECISION_REQUIRED` — báo nguyên văn + phương án, chờ user;
- package/API thật lệch adapter mà sửa thì đổi methodology;
- checksum/data mismatch (kể cả khi quên `git lfs pull`), LF không dẫn xuất từ đúng HF (LF_DERIVATION_MISMATCH) hoặc không phủ HF; lock-s0 báo overlap bất thường với cột S0;
- phát hiện leakage hoặc bug correctness mới; test fail mà sửa thì phải đổi methodology; ERROR trong checker_log không sửa được bằng env/code;
- OOM/hết dung lượng không xử lý được bằng tinh chỉnh execution an toàn (giảm gpu_slots_per_device / P0_GPU_DEVICES, tail_bars, dọn cache HF).

KHÔNG phải lý do để dừng: một bước/model vừa xong; champion đổi/không đổi; ETA cao; cần commit/push.

=========================== BÁO CÁO ===========================
Cuối mỗi bước và cuối run: lệnh, thời gian thật vs ETA, GPU, file output, số liệu chính (MedianGain/WinRate/P10/Worst, ε,
số vòng/epoch LoRA, KEEP/DROP, prune, win, champion đổi/giữ ở replay, TFM-final = hệ thống A hay B), thời gian bận của TỪNG GPU (scheduler_log), bất thường, việc kế tiếp, commit hash.
Bắt đầu bằng: đọc các file context ở trên, `git log --oneline -1`, rồi chạy preflight.
