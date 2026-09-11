# Vast session — TimesFM / AutoTS, BTC 1m hai năm

Đây là runbook đang có hiệu lực trên branch `tfm_autots`. Prompt `/goal` ngắn ở `docs/VAST_GOAL.txt`.
Goal bao gồm FULL TRAINING, summaries/report và push; không dừng ở setup, cache hoặc một model xong.
Không chạy `src_OB`, không tìm/crawl Order Book. Tài liệu OB trước đó đã archive.

## 1. Clone và dữ liệu

Làm trên instance Vast đã được cấp. Nếu đã có đúng checkout thì tiếp nối, không clone đè/reset.
Nếu chưa có, ở thư mục làm việc có đủ disk:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch tfm_autots --single-branch https://github.com/tson295/P0_forecasting.git
cd P0_forecasting
git lfs install --local
git lfs pull --include="data/BTC_1m_2y.csv,data/BTC_5m_2y.csv" --exclude=""
```

Cài `git-lfs` nếu thiếu. Clone skip smudge để không tải toàn bộ LFS experiments cũ/prepared OB.
Feature definitions S0 dùng JSON/CSV có sẵn trong `experiments/15d/` (`b0_star`, wins, keepdrop),
không cần nạp toàn bộ figures/checkpoints/predictions cũ. Không stage các thay đổi LFS ngoài scope.

Đọc `.claude/CLAUDE.md`, `.claude/MEMORY.md`, `.claude/AGENT.md`, checker, `docs/TFM_AUTOTS_PHASE.md`,
`docs/TFM_AUTOTS_VAST_REVIEW.md` và `configs/tfm_autots.json`. Chỉ mở code liên quan nếu có lỗi cụ thể,
không tái điều tra toàn repo hay tự dựng pipeline mới.

Data canonical theo manifest đã commit:
- HF 1m: 1.051.201 bar, 2024-09-03 16:29 → 2026-09-03 16:29 UTC, 101.766.374 bytes,
  SHA256 `559ce040efd737d38f6d541b26e1533f4afc4682b5af2a94ef5cf842e31f8097`.
- LF 5m: 210.239 bar dẫn xuất cho features 5m, 21.146.273 bytes,
  SHA256 `0e5fb9ad20478dd4cc8b26c3669453ff35b3ceff1c72b81d20a6337540f52fef`.
- Giữ `data/data_checksums_2y.json` và `data/BTC_5m_2y.derivation.json`. CLI đối chiếu chúng trong lần load thật.
  LFS pointer chưa phải CSV thật; thiếu data thì sửa download/auth, không ghi checksum mới để lách lỗi.

## 2. Môi trường GPU

Được cài/build dependencies trong instance hiện có. Đọc GPU/driver, RAM/disk và package/build metadata;
không thử fit/probe/infer để chọn backend. Một process training trên một GPU, config mặc định GPU 0.
Không gộp VRAM; không thuê/đổi GPU hoặc giảm batch/epochs/context/search space/validations.
Reuse env tương thích nếu đã có; không force-reinstall/build lại chỉ vì session mới.

Env mới: dùng Python phù hợp package (ưu tiên Python 3.11 nếu có), venv `.venv`, CUDA toolkit/NCCL shared,
compiler/CMake phù hợp image. Các lệnh cài đặt tham chiếu cho image CUDA 12.8:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-tfm-autots.txt
.venv/bin/python -m pip install --no-binary=lightgbm --no-cache-dir \
  --config-settings=cmake.define.USE_CUDA=ON \
  --config-settings=cmake.define.BUILD_WITH_SHARED_NCCL=ON lightgbm
```

Cài toolchain hệ thống nếu thiếu và đã có quyền root/sudo. Nếu có CPU LightGBM sẵn, phải rebuild CUDA;
không coi `pip install` báo satisfied là đã build GPU. Có thể chỉ định CMAKE_CUDA_ARCHITECTURES theo GPU thật
để tránh build các arch không dùng. Dùng shared NCCL để tránh lỗi nvlink ABI của static NCCL.
Không cài JAX/XReg; giữ autots==1.0.4 và timesfm[torch]==2.0.2. Không gọi `scripts/vast_bootstrap.sh`.
Nếu image khác CUDA 12, chuẩn bị bộ CUDA torch/CuPy/build tương thích và ghi lại dependency changes;
không đổi backend thành CPU. Checkpoint TimesFM pinned tự tải khi actual model load, để ngoài experiments.
Metadata thấy GPU chưa chứng minh fit GPU đã thành công; bằng chứng fit lấy từ training thật.

## 3. Chạy ngay full phase

Sau setup, checker đọc config/data/env evidence; lỗi có bằng chứng thì session chính sửa.
Không biến checker thành vòng xin duyệt. Không test/smoke/canary/pytest/probe fit/synthetic run,
check-data riêng, benchmark, latency replay hoặc warmup. Không chạy lệnh `orchestrate`, `final`,
champion/ensemble hoặc models khác. S0/caches chuẩn bị trong đường chạy thật, không phải test.

Chạy launcher trong tmux từ repo (nếu tmux session đã tồn tại, đọc trạng thái rồi attach, không chạy trùng):

```bash
tmux new-session -d -s p0_tfm_autots 'bash scripts/vast_tfm_autots_run.sh'
```

Launcher dùng `.venv/bin/python`, ghi package versions/GPU/config/git/log vào
`experiments/tfm_autots_sessions/run_*`, giữ khóa process và gọi:
`P0_TFM_AUTOTS_VAST=1 python -u run.py tfm-autots --config configs/tfm_autots.json`.
Tự thêm `--resume` khi có phase_progress.json. Output phase: `experiments/tfm_autots/`.
Không đặt file log/setup/report trong output phase trước lần chạy đầu: guard yêu cầu output mới rỗng.
Nếu cần config recovery riêng chỉ để đổi experiments_dir, truyền đường dẫn đó cho launcher, ghi rõ lý do.

CLI chạy S0 → loop tfm → tfm-final → loop autots_wr → loop autots_mr → autots-search → summary.
Giữ 163-column candidate pool; số candidate thực tế theo S0 collision handling. Giữ 5 folds/seeds trong config.
Cache phải READY trước calibration/candidates. TimesFM forecast-first/residual suffix causal; không native XReg.
Từng seed/fold/epoch mode vẫn có LoRA fit riêng; MR recursive steps và native final bake-off vẫn có CPU work.
Không kết luận treo chỉ vì chưa sang candidate khi epoch/cache/validation log còn tiến triển.

Theo dõi tmux, process, log thật và nvidia-smi định kỳ, cập nhật MEMORY. Thời gian dài không phải lý do dừng.
Trước compact ghi config, stage, log, tmux/process và next step, sau compact tiếp tục cùng goal.
Khi lỗi runtime/env/API: giữ traceback, sửa đúng nguyên nhân và tiếp tục, không benchmark để xác nhận.
Không chạy lại cell/stage đã hoàn tất chỉ để đo thêm latency. `--resume` chưa có optimizer resume giữa epoch
hay từng validation của bake-off; giữ failed attempt và chỉ chạy lại phần chưa hoàn tất mà CLI cho phép.
Code/config đổi sẽ bị contract guard từ chối reuse: không sửa hash/bỏ guard. Phân loại ảnh hưởng và giữ
artifact cũ, bổ sung recovery có provenance nếu cần, không tự gọi tất cả kết quả cũ là còn hợp lệ.

## 4. Checker, báo cáo và push

Checker chỉ đọc evidence sau setup, khi có lỗi cụ thể và cuối phase; không trước/sau từng candidate.
Session chính điều phối và sửa code/env. Không agent researcher/runner tự quyết feature/model/hyperparameter.

Đối chiếu đủ stage/candidate/fold/seed thực tế, wins/tfm và wins/autots, seed predictions, calibration,
keepdrop/prune, templates, adapters/estimators và runtime/cache provenance. Không dùng checklist 96 cells OB.
Summary phải có `tfm_autots_per_fold_horizon.csv` và `tfm_autots_summary.csv` từ đúng final representatives.
RMSE và E0 gains ở raw price; không gọi correlation r là R². Metric/latency nào chưa xuất thì ghi rõ,
không bịa hoặc thêm inference benchmark. Batch p95 không phải single-request p95; max không phải hard bound.

Ghi RUN_REPORT.md trong output sau khi phase bắt đầu: data coverage thực, stages/candidates, kết quả,
runtime từng phần, GPU/package/code/config provenance, lỗi đã sửa, hạn chế và trạng thái push.
Ghi CHECKER_FINDINGS.md, cập nhật MEMORY đúng evidence. Code/launcher chưa được run ở phiên local chuẩn bị.

Dùng GitHub credentials đã cấp (SSH agent, credential helper hoặc GH_TOKEN qua gh auth setup-git),
không in token, nhét token vào URL/log hoặc commit secret. Không có credential thì ghi PUSH_PENDING,
commit local và tiếp tục training/report, không ngồi chờ user. Không thể hứa push thành công khi thiếu quyền.
Sau stage hoàn tất và cuối goal, stage rõ các file code/config/docs/.claude đã sửa cùng artifact hoàn tất
trong output phase và session logs. Không `git add -A`, không force push, không sửa `experiments/15d`.
Không add file đang ghi dở, venv hoặc checkpoint pretrained gốc; binary experiments dùng LFS đã cấu hình.
Không thêm ignore metric/predict/checkpoint/cache. Push bằng `git push origin tfm_autots`.
Nếu remote đã tiến: fetch, reconcile trên lịch sử hiện có, giữ local work, không reset/force push.
Nếu auth/quota thật sự chặn: giữ commits/LFS objects và ghi bước còn thiếu; backup trên cùng disk chưa phải off-instance backup.

COMPLETE chỉ khi full phase hợp lệ, đủ artifacts/summaries/reports/checker và push thành công.
Nếu blocker không sửa được trong quyền hiện có, báo BLOCKED/PUSH_PENDING với evidence và phần còn thiếu;
không báo COMPLETE giả. Không kết thúc goal chỉ vì data, cache hoặc một model vừa xong.
