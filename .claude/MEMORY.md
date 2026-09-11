PHASE: TFM_AUTOTS — BTC 1m two years
TRAINING: NOT_RUN_IN_LOCAL_PREPARATION

Branch tfm_autots. Task hiện tại: chuẩn bị code và prompt để chạy một goal trên Vast.
Chưa chạy tests/smoke/probe/training/inference/benchmark trong phiên chuẩn bị.
Đã đọc source/manifest, chưa có runtime mới để chứng nhận sạch lỗi hoặc ước lượng ETA.

Data: BTC_1m_2y.csv qua Git LFS, 1.051.201 bar theo manifest,
2024-09-03 16:29 → 2026-09-03 16:29 UTC. BTC_5m_2y.csv chỉ hỗ trợ feature 5m.
Config: configs/tfm_autots.json, output experiments/tfm_autots, previous feature definitions experiments/15d.

Đã có: TimesFM-first LoRA + held-out residual heads, batch/native forecast cache;
CPU feature pools cho TFM/AutoTS trước candidate; WR/MR cached train design và prediction histories;
LoRA loss bookkeeping trên device và epoch logging; AutoTS-final reuse confirmation selection seed.
Launcher scripts/vast_tfm_autots_run.sh: khóa chống chạy trùng, env/log riêng, tự chọn resume khi có progress.
Đọc docs/TFM_AUTOTS_VAST_REVIEW.md để biết phần chi phí và recovery vẫn còn.

Exact next step trên Vast: docs/VAST_GOAL.txt → clone đúng branch → docs/VAST_SESSION_PROMPT.md →
LFS + env CUDA → launcher trong tmux → theo dõi đến hết phase → checker/report → commit/push.
Không chạy bootstrap/gpu-probe/check-data riêng hoặc pipeline OB. Không giảm workload.
Session chính điều phối, checker chỉ đọc evidence. Cập nhật MEMORY bằng trạng thái run thật trước compact.
Ghi chú OB trước đây đã archive ở docs/archive/ob_before_tfm_vast_2026-09-11/.
