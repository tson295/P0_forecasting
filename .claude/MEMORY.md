PHASE: OB — Vast 2026-09-11: HF pinned BLOCKED; user chọn Zenodo 20046390 (21 ngày) + fold FIT9/gap1/VAL2/step2
TRAINING: RUNNING — Zenodo FULL train bắt đầu 2026-09-11T02:46:25Z, tmux `ob:ztrain`, kỳ vọng 5×8×3 = 120 cell

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
- Recovery: `train.py` `mkdir(exist_ok=False)` + dừng cả run khi một cell lỗi; CLI chưa có `--resume`. Nếu lỗi: lưu
  traceback, sửa + commit, thêm skip cell completed / lưu attempt mới có provenance, chạy phần còn lại bằng
  `--models/--folds` với cùng launcher env (HF_HOME!). Không ghi đè cell completed.

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
