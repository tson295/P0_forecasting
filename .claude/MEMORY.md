PHASE: OB — Vast 2026-09-10: HF pinned BLOCKED; tìm nguồn thay thế ~46 phút (3 vòng, 27 nhóm) không có nguồn phù hợp; replay v2 (W1) đã sửa
TRAINING: NOT_STARTED — BLOCKED_BY_DATA (0 cell); chờ user quyết định nguồn data/phương pháp

## Current task

Goal tiếp nối (`docs/VAST_SESSION_PROMPT.md` bản 109cbc2, `docs/VAST_GOAL.txt`): lưu việc → sửa W1 → tìm data thay thế
→ prepare/train. Đã làm hết phần được phép; kết thúc BLOCKED vì không có nguồn Binance Spot L2 miễn phí/đã có quyền nào
đáp ứng phương pháp cố định (instance/repo cũng không có credential nhà cung cấp dữ liệu nào). Không train. Không
smoke/canary/test/trial fit.

## Trạng thái có bằng chứng (Vast C.50503596, RTX 3090 24 GB, driver 595.84, image CUDA 12.8)

- Instance KHÔNG có volume. Venv `/home/ubuntu/venv-ob` (`experiments/orderbook_hf/run_meta/install_env.sh`,
  LightGBM 4.7.0 CUDA cần `BUILD_WITH_SHARED_NCCL=ON`). GPU env đã cài, chưa có fit nào.
- Backup: SSH key `~/.ssh/id_ed25519` push được `origin/OB` (`remote.origin.pushurl` = SSH). Mọi commit của lượt tiếp nối đã
  push kèm LFS; commit mới nhất xem `git ls-remote origin refs/heads/OB` và `experiments/orderbook_hf/BACKUP_STATUS.md`.
- HF `873f31e7…`: depth 1.392 run ≈300 s (5 phút/giờ, lỗi flush-overwrite collector v1), 1/38 snapshot nối được;
  prepared `data/orderbook/prepared_hf/` (schema v3, replay v1): segment tốt nhất 300,6 s, 0 origin FIT/VAL.
  Raw Parquet không commit (tải lại được).
- Replay v2 (`src_OB/reconstruct.py`, `REPLAY_VERSION = 2`): buffer bridge trước snapshot, không neo lại snapshot cũ khi
  book sống (redundant chỉ khi message kế tiếp được chấp nhận), buffer sống qua anchor lỗi, ID check trước timestamp, hard
  gap theo `known_hard_gaps_utc` trong config và không bridge qua hard gap, mỗi timestamp snapshot chỉ thử neo một lần (bản mới
  nhất trước). Manifest prepared ghi `code_commit`/`code_uncommitted_paths`/`config_sha256` (F-02); `Data` cho train từ
  chối prepared có `replay_version` khác replay hiện tại hoặc thiếu `code_commit` (R6-W1/R7-I3) ⇒ `prepared_hf` (replay v1)
  không train được; prepare mới phải chạy trong git checkout. **Chưa có prepare thật nào dùng
  v2**; prepared HF v3 giữ nguyên (checker suy luận từ evidence, chưa chạy thật: trên HF v2 chỉ đổi nhãn reset). Các lượt
  checker không có ERROR. Còn mở trước prepare v2 thật đầu tiên: R3-I1, R3-I2 (xem CHECKER_FINDINGS).
- Nguồn đã xét: `experiments/orderbook_hf/SOURCE_REPORT.md` (27 nhóm, evidence trong `run_meta/source_search/`). Gần nhất:
  HF `predict-quant/binance-spot-orderbook` @bf8ffb20… (Spot diff 100 ms, run dài nhất 22,64 h < 25,6 h, VAL ≤ 7,5 h,
  không license), Zenodo 20046390 (Spot 5 s, 21 ngày < 30, phi thương mại), HF `Lazy108/binance-polymarket-orderflow`
  (gated manual, 22 ngày). Tardis không key chỉ ngày đầu tháng.
- Reports: `experiments/orderbook_hf/{DATA_REPORT,RUN_REPORT,SOURCE_REPORT,CHECKER_FINDINGS,BACKUP_STATUS}.md`.

## Quyết định còn cần từ user (danh sách chuẩn: RUN_REPORT §5)

1. Nguồn Binance Spot L2 có khóa/trả phí hoặc quyền truy cập (Tardis key, Binance historical data access, Crypto Lake/Kaiko,
   duyệt dataset gated…) ≥ 30–58 ngày liên tục; hoặc
2. đổi phương pháp tường minh (vd. FIT ngắn hơn cho bộ Zenodo 21 ngày, hoặc context TimesFM/h180 khác cho predict-quant);
   hoặc
3. thu thập mới ≥ 30 ngày (1 fold) / ≥ 58 ngày (5 fold) bằng collector liên tục.

## Exact next step khi có quyết định

Đóng R3-I1/R3-I2 → config riêng cho nguồn mới (`dataset_*`, `raw_dir`, `prepared_dir`, `output_dir` mới;
`known_hard_gaps_utc` chỉ khi nguồn công bố) + downloader/adapter nếu schema khác → download → prepare (replay v2) →
`data-report` → checker → nếu READY: `P0_OB_VAST=1 CUDA_VISIBLE_DEVICES=0 python -m src_OB train --config <cfg>` trong tmux
→ `summarize` → checker → báo cáo. CLI vẫn chưa có `--resume`/`--horizons`.

## Nguồn đang có hiệu lực

`src_OB/README.md`, `configs/orderbook.json`, `.claude/CLAUDE.md`, `.claude/AGENT.md`,
`.claude/agents/checker.md`, `docs/VAST_SESSION_PROMPT.md`, `docs/VAST_GOAL.txt`. `docs/IDEA.md` cung cấp ý tưởng,
không thay yêu cầu Direct hiện tại. Memory/agent/prompt OHLCV cũ nằm trong `docs/archive/claude_legacy_2026-09-10/`.
