PHASE: TFM_AUTOTS — implementation only; runtime unverified
TRAINING: NOT_RUN_IN_THIS_SESSION

Branch `tfm_autots`: chỉ `tfm`, `autots_wr`, `autots_mr`; hai nhánh AutoTS gộp thành AutoTS-final.
TimesFM forecast trước, residual heads học trên held-out suffix của FIT với features + forecast vector.
WR/MR predict batch; xem `docs/TFM_AUTOTS_PHASE.md`. Không tests/smoke/probe/benchmark/training local.
Lệnh cho lượt chạy thật sau này trên Vast: `P0_TFM_AUTOTS_VAST=1 python run.py tfm-autots`.
Output riêng `experiments/tfm_autots`; không reuse kết quả residual-first cũ.
Các ghi chú OB dưới đây là lịch sử của nhánh trước, không phải next step của phase này.

## Current task

Thực hiện goal `docs/VAST_SESSION_PROMPT.md` khi user đưa prompt vào session trên Vast:
hoàn thiện data thật → training đủ family/fold/horizon → summaries → checker/report.
Không cần nhiều agent; chỉ checker đọc evidence. Không smoke/canary/test/trial fit.

## Trạng thái có bằng chứng

- Branch `OB`, repo `https://github.com/tson295/P0_forecasting`.
- `src_OB` đã viết, code cũ `src/p0` được giữ. Chưa chạy prepare/replay, training, inference hoặc test trên pipeline mới.
- Raw HF đã tải thành công ở workspace local: `data/orderbook/hf_crypto_lob_stream/`.
  4 Parquet BTCUSDT tháng 6 và 7/2026, tổng **704.186.850 byte** (depth + snapshots).
- Revision dataset: `873f31e729ae23b1c309cd5dcb33feed27c407de`.
  `download_manifest.json` đã được track. **Bốn raw Parquet chưa được commit/push**;
  clone trên Vast chưa có data chỉ vì thấy manifest. Chạy downloader hoặc chuyển đúng các file này.
- Prepared dự kiến: `data/orderbook/prepared_hf/`, schema v3. Chưa có coverage/segment thực tế.
- Output dự kiến: `experiments/orderbook_hf/`. Chưa có metric, thời gian training hay latency thực đo.

## Những điểm phải xử lý bằng lượt chạy thật

- Correctness và khả dụng của reconstruction với collector cũ; khoảng lịch thực tế không chứng minh liên tục.
- TimesFM context 512 điểm cách nhau h giây: ở h=180 cần riêng **25 giờ 33 phút** context cùng segment,
  còn phải có label. Snapshot reanchor/gap làm segment ngắn; common origin mask có thể rỗng cho tất cả model.
  Không nối segment để lách, không khẳng định chỉ còn download data.
- API/dependency AutoTS 1.0.4, TimesFM 2.0.2, LoRA decoder và GPU builds chưa được thực thi trong pipeline OB.
- CLI có `download`, `prepare`, `train`, `summarize`; `--models`, `--folds`. Chưa có `--resume` hay `--horizons`.
  `prepare` yêu cầu thư mục mới; cell output đã tồn tại làm train dừng. Cần recovery có provenance nếu run bị ngắt.

## Exact next step trên Vast

Đọc prompt mới → cài GPU environment không probe/test → download archive pinned → prepare thật → checker đọc
coverage/segment/config → train thật bằng CLI (tự lặp model/fold/horizon) → summarize → checker đọc artifact → report.
Session chính được sửa lỗi env/data/adapter trong scope và tiếp tục, không hỏi unlock giữa bước.
Nếu muốn đổi data revision, context/methodology hoặc tài nguyên ngoài quyền đã cấp, nêu blocker cụ thể.

## Nguồn đang có hiệu lực

`src_OB/README.md`, `configs/orderbook.json`, `.claude/CLAUDE.md`, `.claude/AGENT.md`,
`.claude/agents/checker.md`, `docs/VAST_SESSION_PROMPT.md`. `docs/IDEA.md` cung cấp ý tưởng,
không thay yêu cầu Direct hiện tại. Memory/agent/prompt OHLCV cũ nằm trong `docs/archive/claude_legacy_2026-09-10/`.
