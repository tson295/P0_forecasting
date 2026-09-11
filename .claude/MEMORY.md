PHASE: TFM_AUTOTS — BTC 1m two years
TRAINING: RUNNING_ON_VAST (full phase, bắt đầu 2026-09-11 19:01:35 UTC)

Branch tfm_autots. Instance Vast 1× RTX 3090 24GB (sm_86), driver 580.159.03, không sudo.
Repo /home/ubuntu/P0_forecasting, commit chạy d9201e1 (code_changes.patch rỗng).
Env .venv: Python 3.11.16, torch 2.11.0+cu128, LightGBM 4.7.0 build CUDA (arch 86, shared NCCL hệ thống 2.31.2),
xgboost 3.2.0, cupy-cuda12x 14.2.0, autots 1.0.4, timesfm 2.0.2; không JAX. Chi tiết experiments/tfm_autots_sessions/SETUP_ENV.md.
HF_HOME=/home/ubuntu/.cache/huggingface (vì /workspace/.hf_home root-owned, user ubuntu không ghi được).
Data LFS đã pull, sha256 khớp data/data_checksums_2y.json (1m 559ce0…, 5m 0e5fb9…).

Run: tmux `p0_tfm_autots` (remain-on-exit on) → scripts/vast_tfm_autots_run.sh → python run.py pid 4339.
Log: experiments/tfm_autots_sessions/run_20260911T190135Z_Jmb166/training.log. Output: experiments/tfm_autots/.
Tiến độ thật: lock-s0 xong 19:02:16 (S0_tfm = ∅; S0_wr 72 B0* + 21 ext; S0_mr 72 B0* + 8 ext; 163 candidates/model);
19:02:23 [tfm] chuẩn bị CPU cache trước fit. Chưa có bằng chứng fit GPU thành công.

Exact next step: theo dõi tmux/log/phase_progress.json. Process chết → giữ traceback + run dir, sửa nguyên nhân,
chạy lại launcher (tự --resume). Sửa src/p0 hoặc src_OB/gpu.py làm đổi code hash → phase guard từ chối output cũ:
cần config recovery chỉ đổi experiments_dir, ghi provenance. Stage xong → commit/push output + session logs
(không add -A, không file đang ghi dở). Cuối phase: checker cuối → experiments/tfm_autots/RUN_REPORT.md,
CHECKER_FINDINGS.md → `git push origin tfm_autots` (pushurl SSH đã auth tson295).
Không chạy bootstrap/gpu-probe/check-data riêng/test/smoke hoặc pipeline OB. Không giảm workload.
Ghi chú OB trước đây đã archive ở docs/archive/ob_before_tfm_vast_2026-09-11/.
