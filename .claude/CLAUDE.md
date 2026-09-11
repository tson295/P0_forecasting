# P0_forecasting — phase tfm_autots trên Vast

Chỉ pipeline OHLCV `src/p0`, hai family TimesFM và AutoTS (WR/MR là hai nhánh AutoTS).
Nguồn có hiệu lực: `docs/VAST_SESSION_PROMPT.md`, `docs/TFM_AUTOTS_PHASE.md`,
`docs/TFM_AUTOTS_VAST_REVIEW.md`, `configs/tfm_autots.json` và code entrypoint.
`src_OB`, IDEA Order Book, docs/RESEARCH_PLAN.md và docs/archive là lịch sử, không điều khiển phase này.

## Vai trò và quyền

Session chính điều phối môi trường/process/git, chạy CLI đã có, xử lý lỗi thực tế và giữ một goal đến cuối.
Không dò lại thiết kế pipeline, nghiên cứu feature/model mới hoặc dựng hệ thống agent runner.
Chỉ gọi `checker` để đọc evidence ở mốc cần thiết; Python tự lặp stage/candidate/fold/seed.
Khi user đưa goal Vast: được cài/build dependency GPU, tải đúng Git LFS/pretrained pinned, chạy training thật,
sửa lỗi env/runtime/tích hợp mà giữ phương pháp, ghi artifact, commit/push `tfm_autots` bằng quyền có sẵn.
Không hỏi unlock/approval giữa các bước đã được giao. Thiếu credential không được bịa hoặc lộ token.
Không thuê, đổi, stop/destroy instance. Phiên sửa code local không tự khởi chạy training.

## Bất biến

- BTC 1m hai năm: `data/BTC_1m_2y.csv`, 2024-09-03 16:29 → 2026-09-03 16:29 UTC theo manifest.
  `BTC_5m_2y.csv` chỉ cho features 5m; giữ checksum/sidecar. Lấy bằng Git LFS, không crawl Order Book.
- Config giữ 5 fold rolling_spread: FIT 120d, ES 5d, VAL 3d, purge 60m; giữ TEST 30d chưa đánh giá.
  Không nhầm với gap >5 ngày của phương pháp OB. Giữ seeds, candidates, validation count, batch và epochs.
- TimesFM input r1 gốc → LoRA → forecast batch/cache → GPU residual heads với features + forecast.
  Residual labels học trên suffix FIT chưa dùng để train/chọn LoRA; outer ES/VAL không train adapter/head.
  Không gọi native XReg/JAX; không quay lại residual-first hoặc pooled fit qua tương lai.
- Cache feature pool phải READY trước candidate; scaler FIT-only. MR recursive steps 2–3 vẫn phụ thuộc candidate.
  AutoTS chỉ estimator GPU trong allowlist, native bake-off giữ 10 validations, không genetic CPU search.
- GPU-only training, không CPU fallback. CPU được dùng cho preprocessing/metrics và LightGBM inference.
- Cấm mọi test/smoke/canary/unit/integration/synthetic run/trial fit/probe fit/benchmark/warmup riêng.
  Không gọi bootstrap cũ, gpu-probe, check-data riêng, pytest hoặc workflow tất cả model.
  Được đọc code/data/metadata/log/process/nvidia-smi và đối chiếu artifact của run thật.

## Chạy và khôi phục

Theo `docs/VAST_SESSION_PROMPT.md`: clone branch → LFS → CUDA env →
`bash scripts/vast_tfm_autots_run.sh` trong tmux → checker/report → commit/push.
Launcher chỉ chạy một phase trong checkout, ghi log/env ở `experiments/tfm_autots_sessions/`.
Không tạo log/file sẵn bên trong output phase còn mới (`experiments/tfm_autots/`).
`--resume` giữ progress/caches khi đúng contract, chưa có optimizer resume hoặc recovery mỗi validation AutoTS.
Không lặp mù lệnh lỗi. Giữ traceback và attempt, sửa nguyên nhân; không giảm workload để che lỗi.
Thay code/config khiến contract đổi: giữ artifact cũ, xử lý recovery có provenance; không sửa hash để ép reuse.

## Kết quả và git

Không sửa data canonical, Baseline_LGBM.py, src_OB hoặc experiments/15d trong phase này.
Không ignore artifact. Stage rõ code/config/docs/.claude và thư mục output mới; không git add -A/force push.
Không commit secret, venv, cache pretrained gốc. Binary experiment dùng LFS đã cấu hình.
Push theo stage hoàn tất và cuối goal; không upload file đang ghi dở.
Thiếu auth/quota push: ghi PUSH_PENDING, vẫn hoàn thành training/report được phép; không báo đã backup ngoài máy.
COMPLETE chỉ khi đủ stages, artifacts/summaries/checker report và push thành công; code commit không chứng minh runtime.
Lưu stage, tmux/process, config, log và exact next step trong MEMORY trước compact; tiếp tục cùng goal sau compact.

@MEMORY.md
