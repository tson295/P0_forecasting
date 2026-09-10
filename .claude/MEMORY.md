PHASE: OB — Vast 2026-09-10: download + prepare thật xong; DATA BLOCKED (archive pinned chỉ dựng được 300,6 s book)
TRAINING: NOT_STARTED — BLOCKED_BY_DATA (0/96 cell); chờ user quyết định nguồn data/phương pháp

## Current task

Goal `docs/VAST_SESSION_PROMPT.md` dừng ở bước 4 vì data không khả dụng với phương pháp cố định.
Không train cho tới khi user chọn nguồn data mới (hoặc đổi phương pháp tường minh). Không smoke/canary/test/trial fit.

## Trạng thái có bằng chứng (Vast C.50503596, RTX 3090 24 GB, driver 595.84, image CUDA 12.8)

- Instance KHÔNG có volume: recycle/destroy mất toàn bộ container (raw, venv, prepared). Venv `/home/ubuntu/venv-ob`
  dựng bằng `experiments/orderbook_hf/run_meta/install_env.sh` (LightGBM 4.7.0 CUDA cần `BUILD_WITH_SHARED_NCCL=ON`).
- Raw HF revision `873f31e729ae23b1c309cd5dcb33feed27c407de` tải lại bằng downloader, sha256 4 file khớp manifest.
  Bốn raw Parquet vẫn không commit (tải lại được, 704.186.850 byte).
- Depth raw chỉ có 1.392 run ID-liên tục ≈300 s (5 phút/giờ, phút :49–:55), 115,8 h / 1.391 h. Nguyên nhân upstream:
  lỗi flush-naming ghi đè trong cùng giờ của collector v1 tháng 6–8/2026 (card `Goooddy/crypto-lob-stream` 2026-09-01);
  file 2026-08 của repo nguồn cũng bị. Repo pinned chỉ có 1 commit.
- Prepare `data/orderbook/prepared_hf/` (schema v3): 38 segment, chỉ 1 segment > 1 state
  (2026-07-05 20:50:12 → 20:55:12, 3.177 state, 24 origin); 3.214 raw state, 61 origin; 1/38 snapshot nối được ID.
- `python -m src_OB data-report` (mới; dùng `select_origins` chung với train): 4 fold theo lịch, 0 origin FIT/VAL ở mọi mask.
  Report: `experiments/orderbook_hf/{DATA_REPORT.md,data_report.json,RUN_REPORT.md,CHECKER_FINDINGS.md}`.
- Checker bước 4: không ERROR; B1 blocker xác nhận (VAL rỗng mọi fold kể cả context=1, bỏ TimesFM).
  W1: replay bỏ message bridge nhận trước timestamp snapshot / check ts trước contiguity — sửa khi prepare nguồn mới
  (prepared_dir mới). I1: nhãn `invalid_timestamp_gap` ở 35 segment thực chất cũng là gap ID.
- Chưa có metric, latency, checkpoint, summary. Env GPU đã cài nhưng chưa có fit nào.
- Commit của lượt này chỉ ở local branch OB nếu push không được: instance không có credential GitHub
  (`git push` → "could not read Username"). Cần user push hoặc cấp credential.

## Quyết định còn cần từ user

1. Nguồn historical BTCUSDT Spot depth liên tục khác (vd. Tardis, cần key/chi phí), hoặc
2. thu thập mới bằng collector đã sửa (≥ 30 ngày cho 1 fold, ≥ 58 ngày cho 5 fold) / chờ upstream tháng 9+, hoặc
3. đổi phương pháp tường minh để dùng archive 5 phút/giờ (không khuyến nghị; TimesFM/AutoTS vẫn không khả thi).

## Exact next step khi có quyết định

Dữ liệu mới → config mới (`dataset_*`, `raw_dir`, `prepared_dir` mới, không ghi đè bản cũ) → download → prepare →
`data-report` → checker → nếu READY: `P0_OB_VAST=1 CUDA_VISIBLE_DEVICES=0 python -m src_OB train` trong tmux → summarize
→ checker → RUN_REPORT. CLI vẫn chưa có `--resume`/`--horizons`.

## Nguồn đang có hiệu lực

`src_OB/README.md`, `configs/orderbook.json`, `.claude/CLAUDE.md`, `.claude/AGENT.md`,
`.claude/agents/checker.md`, `docs/VAST_SESSION_PROMPT.md`. `docs/IDEA.md` cung cấp ý tưởng,
không thay yêu cầu Direct hiện tại. Memory/agent/prompt OHLCV cũ nằm trong `docs/archive/claude_legacy_2026-09-10/`.
