PHASE: OB — Vast 2026-09-11: HF pinned BLOCKED; user chọn Zenodo 20046390 (21 ngày) + fold FIT9/gap1/VAL2/step2
TRAINING: COMPLETE — Zenodo 120/120 cell (5 fold × 8 family × 3h), TRAIN_EXIT=0 16:18:14Z, summarize 0; checker cuối
không ERROR. Còn 1 quyết định user: target AutoTS (RUN_REPORT §6)

## Kết quả (có bằng chứng; chi tiết `experiments/orderbook_zenodo/RUN_REPORT.md`)

- Config `configs/orderbook_zenodo.json`; raw `data/orderbook/zenodo_20046390/` (tar md5 `58507a0f…`, không commit);
  prepared `data/orderbook/prepared_zenodo_20046390/` (replay 2, code c68fc45, LFS); output `experiments/orderbook_zenodo/`.
- Không family nào thắng E0 nhất quán (9/120 cell gain > 0). tfm_zero_shot gần E0 nhất (mean gain −0,025…−0,040;
  pooled +0,002/+0,014 ở h120/h180). Cây OF/OFI gần E0 (xgbrf/cat tốt nhất h60). LSTM, LoRA tệ hơn E0 mọi cell
  (LoRA overfit, thua zero-shot 15/15). AutoTS có RMSE 1,8–38× E0: cây hồi quy mức giá thô bị kẹp quanh mức giá FIT,
  không ngoại suy được (giới hạn thiết kế, ZF-W1). Latency p95 batch 1: LSTM 0,34 ms, cat/lgbm ~0,8 ms, xgb ~1,9,
  xgbrf ~2,9, AutoTS ~3, zero-shot ~82, LoRA ~95 ms.
- Sự cố đã sửa (giữ phương pháp): LightGBM GOSS CUDA crash + `linear_tree` âm thầm fit CPU (attempt1, bị bỏ) →
  688fefd; SIGSEGV max_bin 1000 → 2a5a1c3 (`max_bin ≤ 255`). `train --resume` có từ 688fefd. Attempts giữ ở
  `experiments/orderbook_zenodo/attempts/`.
- Checker: Z-pre và ZF không ERROR; WARN Z-W1, ZF-W1, ZF-W2 đã sửa trong docs/report (`CHECKER_FINDINGS.md`).
- Backup: toàn bộ artifact push `origin/OB` (commit 40b4c45 + commit report cuối; xem `BACKUP_STATUS.md`).

## Quyết định còn cần từ user

1. AutoTS: dùng log-price center theo giá origin, target `log(MP(t+h)/MP(t))`, rồi chạy lại 15 cell AutoTS vào
   output/attempt mới có provenance (không ghi đè cell hiện tại, không tune bằng outer VAL). Chưa duyệt thì không làm.

## Exact next step (nếu user duyệt mục 1)

Sửa `src_OB/autots_native.py` (series + target log-relative, giữ allowlist/regressor/fold) → commit → chạy AutoTS vào
output mới, ví dụ config copy với `output_dir` mới và `--models autots`, launcher giữ `HF_HOME`/`P0_OB_VAST=1`/
`CUDA_VISIBLE_DEVICES=0` → summarize → checker → cập nhật report. Không smoke/test/trial fit.

## Môi trường Vast (C.50503596, RTX 3090 24 GB, driver 595.84, CUDA 12.8, không volume)

- Venv `/home/ubuntu/venv-ob` (`experiments/orderbook_hf/run_meta/install_env.sh`; LightGBM 4.7.0 CUDA cần
  `BUILD_WITH_SHARED_NCCL=ON`). HF_HOME phải là `/home/ubuntu/.cache/huggingface` (global trỏ `/workspace/.hf_home` của
  root). Launcher: `experiments/orderbook_zenodo/run_meta/run_train.sh [--resume]`.
- Push: SSH key, `remote.origin.pushurl` = SSH; commit với `-c user.name/-c user.email`; stage theo scope.

## HF (đóng, BLOCKED — chỉ tham khảo)

- HF `873f31e7…`: depth ≈300 s/giờ (collector v1 lỗi), segment tốt nhất 300,6 s, 0 origin. Replay v2 đã sửa (W1),
  chưa prepare thật trên HF; R3-I1/R3-I2 còn mở (chỉ diff replay). Reports `experiments/orderbook_hf/*.md`,
  nguồn đã xét `SOURCE_REPORT.md` (27 nhóm).

## Nguồn đang có hiệu lực

`src_OB/README.md`, `configs/orderbook_zenodo.json` (run hiện tại), `configs/orderbook.json` (HF), `.claude/CLAUDE.md`,
`.claude/AGENT.md`, `.claude/agents/checker.md`, `docs/VAST_SESSION_PROMPT.md`, `docs/VAST_GOAL.txt`.
Memory/agent/prompt OHLCV cũ nằm trong `docs/archive/claude_legacy_2026-09-10/`.
