# Prompt cho session Claude Code trên Vast — vòng EXPANDED-DATA (2026-09-03)

> Chỉ dùng khi user đã **cho phép rõ ràng** chạy experiment vòng expanded-data (MEMORY hiện ghi `TRAINING: LOCKED`).
> Copy nguyên khối dưới đây vào một session Claude Code mới trên máy Vast sau khi: instance đã tạo, repo đã clone vào
> `~/P0_forecasting` (kèm `git lfs install`), và `data/BTC_hf_1min_full.csv` + `data/BTC_lf_5min_full.csv` đã scp lên (CSV không nằm trong git).

---

Bạn là session Claude Code chạy trên máy Vast.ai GPU cho project P0_forecasting (BTC 1-phút point forecasting), vòng
EXPANDED-DATA. Repo ở ~/P0_forecasting; data/BTC_hf_1min_full.csv và data/BTC_lf_5min_full.csv đã được scp lên.

PROMPT NÀY LÀ AUTHORIZATION CỦA USER ĐỂ CHẠY TOÀN BỘ EXPERIMENT END-TO-END (user đã unlock bằng cách gửi prompt này):
sau khi mọi preflight PASS, bạn TỰ chuyển TRAINING: UNLOCKED và chạy LIÊN TỤC theo thứ tự plan §8 tới hết `final` + `visualize`.
KHÔNG hỏi user để duyệt giữa các bước/model. Chỉ DỪNG và hỏi user trong các blocker liệt kê ở cuối prompt.

ĐỌC TRƯỚC (theo thứ tự): .claude/CLAUDE.md → .claude/MEMORY.md → docs/RESEARCH_PLAN.md (rev 10) → README.md → .claude/AGENT.md
→ docs/reference/audit_timesfm_lora.md. docs/archive/ là lịch sử; experiments/15d/ là vòng 15 ngày đã xong (không sửa).

=========================== LUẬT BẤT BIẾN ===========================
1. Training CHỈ trên GPU, cấm CPU, không fallback âm thầm; backend từng thư viện do scripts/vast_bootstrap.sh RESOLVE bằng fit thật
   và ghi vào configs/p0_full.json (LightGBM device_type gpu|cuda → chảy sang AutoTS-WR và template bake-off). xreg của TimesFM
   chạy jax GPU (jax[cuda12]==0.11.1, XLA_PYTHON_CLIENT_PREALLOCATE=false). Thiếu GPU backend cho một model bắt buộc → DỪNG hỏi user.
2. TRAINING lock: CLI tự từ chối khi MEMORY còn LOCKED. TEST chỉ chạm ở `final`, đúng MỘT lần.
3. Mỗi run thuộc đúng một bước §8; không chạy trùng, không để GPU idle.
4. Không thêm model/metric/feature ngoài plan; không sweep hyperparameter; không sửa Baseline_LGBM.py; không đổi luật KEEP/DROP,
   prune (chỉ cột mới), confirmation, champion, ensemble, vai trò seed §1.3, S0_m khoá, C_short. Không sửa test để né failure.
5. TimesFM: LoRA per fold (FIT học, ES chọn epoch, VAL không thấy) → freeze → XReg search trên CÙNG adapter; loss MSE trên ŷ_h;
   không torch.compile; inject sau load_checkpoint; giữ mean head, 1 origin/lời gọi, dịch 1 bar, cộng dồn one-step.
   AutoTS: probe = 2 class cố định từ S0 của nhánh; framework chỉ với initial_template GPU + max_generations=0 trên FIT+ES.
6. data/ read-only. KHÔNG ghi đè data/data_checksums_full.json sau khi đã ghi lần đầu. Không secret vào repo/MEMORY/git.
   experiments/** KHÔNG được ignore: commit + push (LFS cho .npz/.pt) sau mỗi model; adapter LoRA trong experiments/full/lora/.
7. KHÔNG vẽ figure trong bất kỳ bước training nào; figure chỉ sinh bằng `python run.py visualize` sau `final`.

=========================== PREFLIGHT (fail-fast) ===========================
  cd ~/P0_forecasting && git lfs install && tmux new -s p0
  git log --oneline -1
  export P0_FOLD_WORKERS=5 XLA_PYTHON_CLIENT_PREALLOCATE=false
  bash scripts/vast_bootstrap.sh                                    # GPU, pip, timesfm 2.0.2 + autots 1.0.4 + jax[cuda12], build/resolve LightGBM, preflight, unit test
  PYTHONPATH=src:. python scripts/vast_canary.py --config configs/p0_full.json
  PYTHONPATH=src:. python scripts/canary_xreg_gpu.py --config configs/p0_full.json
  python run.py check-data --config configs/p0_full.json --write-checksums   # lần đầu: ghi anchor; in fold rolling + TEST 30 ngày; cần ≥ 90 ngày
  python run.py lock-s0 --config configs/p0_full.json               # S0_m khoá + collision audit + Candidate_m → experiments/full/s0/
  PYTHONPATH=src:. python -m pytest -q -x

check-data phải in: HF ok:true, LF phủ HF, 5 fold + final_TEST đều OK. lock-s0 phải in: 0 cặp identical bất thường (near chỉ báo),
S0: lgbm 72+14, xgb 72+11, cat 72+5, xgbrf 72+12, lstm 72+23, autots_wr 72+21, autots_mr 72+8, tfm 0+0; Candidate_m ≈ 97.
Canary LoRA thời gian thật (SAU khi unlock, TRƯỚC `loop --model lgbm`): `PYTHONPATH=src:. python scripts/canary_lora.py --config configs/p0_full.json`
— 1 fold × 1 epoch × 64 origin, adapter vào thư mục tạm, KHÔNG ghi artifact/log của experiment; ghi thời gian/epoch, VRAM đỉnh,
ms/origin XReg và ETA vào báo cáo. Nếu OOM với 5 worker → giảm P0_FOLD_WORKERS cho bước tfm (không đổi batch LoRA nếu không bắt buộc; ghi rõ khi đổi).

TỰ UNLOCK khi và chỉ khi đủ: commit đúng · bootstrap PASS · hai canary PASS · check-data PASS · lock-s0 PASS · pytest PASS.
Khi đó: sửa .claude/MEMORY.md thành "TRAINING: UNLOCKED", commit, rồi chạy tiếp mà không hỏi lại.

=========================== THỨ TỰ CHẠY (plan §8) ===========================
  python run.py loop --config configs/p0_full.json --model lgbm        # BẮT BUỘC đầu tiên (champion ban đầu §3)
  python run.py loop --config configs/p0_full.json --model xgb
  python run.py loop --config configs/p0_full.json --model cat
  python run.py loop --config configs/p0_full.json --model tfm         # LoRA → freeze → XReg add-one → prune → confirmation (+ native)
  python run.py tfm-final --config configs/p0_full.json               # LoRA+XReg(win) vs native LoRA → TimesFM-final → champion
  python run.py loop --config configs/p0_full.json --model xgbrf
  python run.py loop --config configs/p0_full.json --model autots_wr   # probe (S0 = bộ thắng WR cũ)
  python run.py loop --config configs/p0_full.json --model autots_mr   # probe (S0 = bộ thắng MR cũ)
  python run.py autots-search --config configs/p0_full.json           # framework(F_WR_best) vs framework(F_MR_best) → AutoTS-final
  python run.py loop --config configs/p0_full.json --model lstm
  python run.py ensemble --config configs/p0_full.json
  python run.py final --config configs/p0_full.json                   # TEST đúng một lần; lưu final/*.npz
  python run.py visualize --config configs/p0_full.json               # hậu kỳ: mọi figure từ artifact

Sau mỗi model: cập nhật MEMORY (Current Task / Exact Next Step, thời gian thật vs ETA), `git add -A && git commit && git push`
(LFS cho .npz/.pt), rồi chạy tiếp ngay. `loop --resume` nếu SSH rớt giữa add-one (đọc log.csv + calib/<m>_base.json).

=========================== ĐỌC KẾT QUẢ ===========================
- MedianGain vs E0 chỉ cỡ 0.05–0.3 pp là bình thường; Gain > ~1 pp vs B0/E0 → NGHI LEAKAGE, dừng và gọi agent checker.
- TimesFM-LoRA ≈ E0 là kết quả hợp lệ (r1 gần nhiễu trắng); ES trên ES-partition là chốt chặn.
- KHÔNG chạy lại `final` nếu experiments/full/summary/all_models_test.csv đã tồn tại.

=========================== CHỈ DỪNG VÀ HỎI USER KHI ===========================
- không có GPU backend khả dụng cho một model bắt buộc; package/API thật lệch adapter mà sửa thì đổi methodology;
- checksum/data mismatch, LF không phủ HF, data < 90 ngày; lock-s0 báo identical pair bất thường với cột S0;
- phát hiện leakage hoặc bug correctness mới; test fail mà sửa thì phải đổi methodology;
- OOM/hết dung lượng không xử lý được bằng tinh chỉnh execution an toàn (giảm P0_FOLD_WORKERS, tail_bars, dọn cache HF).

KHÔNG phải lý do để dừng: một bước/model vừa xong; champion đổi/không đổi; ETA cao; cần commit/push.

=========================== BÁO CÁO ===========================
Cuối mỗi bước và cuối run: lệnh, thời gian thật vs ETA, GPU, file output, số liệu chính (MedianGain/WinRate/P10/Worst, ε,
số vòng/epoch LoRA, KEEP/DROP, prune, win, champion đổi/giữ, TFM-final native hay +XReg), bất thường, việc kế tiếp, commit hash.
Bắt đầu bằng: đọc các file context ở trên, `git log --oneline -1`, rồi chạy preflight.
