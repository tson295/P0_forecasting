PHASE: OB — Vast 2026-09-10: HF pinned BLOCKED; tìm nguồn thay thế ~60 phút không có nguồn phù hợp; replay v2 (W1) đã sửa
TRAINING: NOT_STARTED — BLOCKED_BY_DATA (0 cell); chờ user quyết định nguồn data/phương pháp

## Current task

Goal tiếp nối (`docs/VAST_SESSION_PROMPT.md` bản 109cbc2, `docs/VAST_GOAL.txt`): lưu việc → sửa W1 → tìm data thay thế
→ prepare/train. Đã làm hết phần được phép; kết thúc BLOCKED vì không có nguồn Binance Spot L2 miễn phí nào đáp ứng
phương pháp cố định. Không train. Không smoke/canary/test/trial fit.

## Trạng thái có bằng chứng (Vast C.50503596, RTX 3090 24 GB, driver 595.84, image CUDA 12.8)

- Instance KHÔNG có volume. Venv `/home/ubuntu/venv-ob` (`experiments/orderbook_hf/run_meta/install_env.sh`,
  LightGBM 4.7.0 CUDA cần `BUILD_WITH_SHARED_NCCL=ON`). GPU env đã cài, chưa có fit nào.
- Backup: SSH key `~/.ssh/id_ed25519` push được `origin/OB` (`remote.origin.pushurl` = SSH). `a04d17b` đã push kèm LFS;
  xem `experiments/orderbook_hf/BACKUP_STATUS.md` và `git ls-remote` cho commit mới nhất.
- HF `873f31e7…`: depth 1.392 run ≈300 s (5 phút/giờ, lỗi flush-overwrite collector v1), 1/38 snapshot nối được;
  prepared `data/orderbook/prepared_hf/` (schema v3, replay v1): segment tốt nhất 300,6 s, 0 origin FIT/VAL.
  Raw Parquet không commit (tải lại được).
- Replay v2 (`src_OB/reconstruct.py`, `REPLAY_VERSION = 2`): buffer bridge trước snapshot, không neo lại snapshot cũ khi
  book sống, ID check trước timestamp, hard gap theo `known_hard_gaps_utc` trong config. Chưa có prepare thật nào dùng v2;
  prepared HF v3 giữ nguyên (checker lượt 2 xác nhận v2 chỉ đổi nhãn reset trên HF). Checker lượt 2: không ERROR;
  T-W1 (snapshot redundant chỉ khi message kế tiếp được book sống chấp nhận) và T-I3 (buffer sống qua anchor lỗi) đã sửa.
  Checker lượt 3 (`1e2f3ce`): không ERROR/WARN; INFO còn mở R3-I1..I5 (xem CHECKER_FINDINGS) — xử lý cùng adapter nguồn mới
  trước prepare thật đầu tiên dùng replay v2.
- Nguồn đã xét: `experiments/orderbook_hf/SOURCE_REPORT.md` (22 nhóm). Gần nhất: HF `predict-quant/binance-spot-orderbook`
  @bf8ffb20… (Spot diff 100 ms, run dài nhất 22,64 h < 25,6 h, không license) và Zenodo 20046390 (Spot 5 s, 21 ngày < 30).
  Tardis không key chỉ ngày đầu tháng. Evidence: `experiments/orderbook_hf/run_meta/source_search/`.
- Reports: `experiments/orderbook_hf/{DATA_REPORT,RUN_REPORT,SOURCE_REPORT,CHECKER_FINDINGS,BACKUP_STATUS}.md`.

## Quyết định còn cần từ user

1. Nguồn Binance Spot L2 có khóa/trả phí (Tardis key, Binance historical data access, Crypto Lake/Kaiko…) ≥ 30–58 ngày
   liên tục; hoặc
2. đổi phương pháp tường minh (vd. FIT ngắn hơn cho bộ Zenodo 21 ngày, hoặc context TimesFM/h180 khác cho predict-quant);
   hoặc
3. thu thập mới ≥ 30 ngày (1 fold) / ≥ 58 ngày (5 fold) bằng collector liên tục.

## Exact next step khi có quyết định

Config riêng cho nguồn mới (`dataset_*`, `raw_dir`, `prepared_dir`, `output_dir` mới; `known_hard_gaps_utc` chỉ khi nguồn
công bố) + downloader/adapter nếu schema khác → download → prepare (replay v2) → `data-report` → checker → nếu READY:
`P0_OB_VAST=1 CUDA_VISIBLE_DEVICES=0 python -m src_OB train --config <cfg>` trong tmux → `summarize` → checker → báo cáo.
CLI vẫn chưa có `--resume`/`--horizons`.

## Nguồn đang có hiệu lực

`src_OB/README.md`, `configs/orderbook.json`, `.claude/CLAUDE.md`, `.claude/AGENT.md`,
`.claude/agents/checker.md`, `docs/VAST_SESSION_PROMPT.md`, `docs/VAST_GOAL.txt`. `docs/IDEA.md` cung cấp ý tưởng,
không thay yêu cầu Direct hiện tại. Memory/agent/prompt OHLCV cũ nằm trong `docs/archive/claude_legacy_2026-09-10/`.
