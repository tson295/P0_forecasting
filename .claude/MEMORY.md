PHASE: OB — Vast 2026-09-11: HF pinned BLOCKED; user chọn Zenodo 20046390 (21 ngày) + fold FIT9/gap1/VAL2/step2
TRAINING: RUNNING — Zenodo FULL train; lần 2 (`--resume`, commit 688fefd) từ ~02:58Z trong tmux `ob:ztrain2`; kỳ vọng 120 cell

## Current task

Goal tiếp nối (`docs/VAST_SESSION_PROMPT.md`, `docs/VAST_GOAL.txt`) + quyết định user 2026-09-11 trong session:
dùng Zenodo (OF giữa snapshot ~1,24 s; nghiên cứu phi thương mại), FIT 9 d / gap 1 d / VAL 2 d / step 2 d / 5 fold
(gap chỉ cần > horizon dài nhất 180 s). Việc còn lại: train xong → `summarize` (launcher tự chạy khi train exit 0) →
checker cuối run → `experiments/orderbook_zenodo/{RUN_REPORT,CHECKER_FINDINGS,BACKUP_STATUS}.md` → MEMORY → commit/push
scope OB (LFS predictions/checkpoints) → `git ls-remote` xác minh. Không smoke/canary/test/trial fit.

## Run Zenodo (có bằng chứng)

- Config `configs/orderbook_zenodo.json`; raw `data/orderbook/zenodo_20046390/` (tar md5 `58507a0f…`, không commit);
  prepared `data/orderbook/prepared_zenodo_20046390/` (replay_version 2, code_commit c68fc45, LFS, đã push);
  output `experiments/orderbook_zenodo/`. Adapter `src_OB/snapshots.py` (SNAPSHOT_ADAPTER_VERSION 1).
- DATA_REPORT READY: 2 segment (chính: 2023-10-01 00:00:06.9 → 10-21 23:59:59.9), 1.439.157 state, 237.648 origin;
  5 fold VAL 10-11→10-21; train/val origins 83671/17447, 80438/7621, 66173/31161, 75100/25130, 79968/31388.
- Launcher `experiments/orderbook_zenodo/run_meta/run_train.sh` (P0_OB_VAST=1, CUDA_VISIBLE_DEVICES=0,
  HF_HOME=/home/ubuntu/.cache/huggingface vì /etc/environment trỏ /workspace/.hf_home của root). Log `logs/train.log`,
  `logs/summarize.log`. Train code = commit d8797c5 (tree sạch lúc start; không sửa `src_OB/*.py` khi đang chạy).
- Checker trước train: không ERROR; Z-W1 (docs luật gap cũ) đã sửa; Z-I1…I8 cần ghi vào RUN_REPORT (cadence thật
  ~1,24 s khác datacite 5 s, timestamp là đồng hồ collector; ngày 10-01 bất thường nhưng ngoài mọi fold; FIT hiệu dụng
  ~7,9 d do context TimesFM h180; fold2 VAL ít origin → nêu mean-fold và pooled; 2 định nghĩa config sha).
- Lần 1 (02:46:25Z, code d8797c5): 15 cell completed (fold1 lgbm/xgb/cat/xgbrf/lstm × 3h; lgbm model text xác nhận
  device_type cuda). Dừng ở fold1/autots/h60s: candidate LightGBM GOSS → `[CUDA] invalid argument goss.hpp 66`
  (LightGBM 4.7.0 không cấp phát CUDA bag buffer cho GOSS). Replay seed cho thấy candidate 3 của attempt đó là
  `linear_tree=True` ⇒ LightGBM đã âm thầm fit CPU (config.cpp:426-430) — attempt bị bỏ, phải ghi trong RUN_REPORT.
  Sửa (688fefd): adapter đặt linear_tree=False, GOSS→gbdt; `GPURegressor` kiểm config hiệu lực sau mỗi fit LightGBM;
  `train --resume` skip cell completed cùng config/revision, chuyển attempt dở sang `attempts/<fold>/<model>/<h>/attemptN`.
- Lần 2 (02:58:19Z, `--resume`, 688fefd): skip đúng 15 cell, attempt1 chuyển sang `attempts/`; candidate 3 và 6 (8-bit
  bin) fit CUDA xong; candidate 9 (LightGBM max_bin 1000) ⇒ SIGSEGV, TRAIN_EXIT=139 (không traceback). `failed.json`
  hậu kiểm ghi vào cell trước khi resume chuyển thành attempt2. Sửa: `max_bin ≤ 255` trong allowlist, launcher bật
  `PYTHONFAULTHANDLER=1`.
- Lần 3 (03:02:28Z, `--resume`, 2a5a1c3): qua được AutoTS; fold1/autots/h60s completed (run.json train_code 2a5a1c3,
  prior_attempts 1+2). Kết quả outer VAL RMSE 498,9 so với E0 15,14: VAL fold1 thấp hơn toàn bộ khoảng giá FIT, và
  adapter hồi quy mức giá thô bằng model cây nên không ngoại suy được. Đây là giới hạn thiết kế, không phải lỗi runtime;
  đã ghi vào RUN_REPORT §3b và là quyết định cho user (đổi target AutoTS sang log-return quanh giá origin rồi chạy lại
  15 cell AutoTS). KHÔNG tự đổi. Candidate dart trên CUDA có smape 200 (dự báo ≈ −28k), bị loại.
- Nhịp đo được: fold1 trees+lstm ~2 phút, autots ~3 phút/cell, tfm_zero_shot ~3 phút/cell (RMSE ≈ E0),
  tfm_lora ~9,3 phút/epoch → ~50 phút/cell (GPU ~61%, 1 lõi CPU 100%: TimesFM chạy fwd/bwd bị giới hạn bởi overhead launch).
  ETA toàn run ~14 h từ 03:28Z (15 cell LoRA ≈ 12,5 h). Không chạy fold song song trên cùng GPU vì làm hỏng
  latency p95/p99; không đổi epoch/batch/context.
- Recovery tiếp: `bash experiments/orderbook_zenodo/run_meta/run_train.sh --resume` (launcher truyền args, giữ HF_HOME).
  Cell mới ghi `train_code` (commit) + `prior_attempts`. Không ghi đè cell completed.

## HF (đóng, BLOCKED — chỉ tham khảo)

- Instance Vast C.50503596, RTX 3090 24 GB, không volume. Venv `/home/ubuntu/venv-ob`
  (`experiments/orderbook_hf/run_meta/install_env.sh`, LightGBM 4.7.0 CUDA cần `BUILD_WITH_SHARED_NCCL=ON`).
- Backup: SSH key push `origin/OB` (`remote.origin.pushurl` = SSH); identity -c user.name/user.email mỗi commit.
- HF `873f31e7…`: depth ≈300 s/giờ (collector v1 lỗi), segment tốt nhất 300,6 s, 0 origin. Replay v2 đã sửa (W1),
  chưa prepare thật trên HF; R3-I1/R3-I2 còn mở (chỉ diff replay). Reports `experiments/orderbook_hf/*.md`,
  nguồn đã xét `SOURCE_REPORT.md` (27 nhóm).

## Nguồn đang có hiệu lực

`src_OB/README.md`, `configs/orderbook_zenodo.json` (run hiện tại), `configs/orderbook.json` (HF), `.claude/CLAUDE.md`,
`.claude/AGENT.md`, `.claude/agents/checker.md`, `docs/VAST_SESSION_PROMPT.md`, `docs/VAST_GOAL.txt`.
Memory/agent/prompt OHLCV cũ nằm trong `docs/archive/claude_legacy_2026-09-10/`.
