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
19:02:54 [tfm] CACHE READY (163 cột × 5 folds, 581 MB). 19:03:5x LoRA real fit đầu (ES, fold1, seed 8586):
158230 windows, batch 64, ≤20 epochs; epoch 1 xong 19:19:02 (~15 phút/epoch, train_mse 2.009e+00, es_mse 2.158e+00),
GPU 73–97 %, 5.2–5.8 GB VRAM. Fit LoRA đầu HOÀN TẤT 20:29 (bằng chứng lora/*.json + runtime/*.json):
ES fold1 seed 8586, 6 epochs, best_epoch 1 (es_mse 2.158 rồi tăng), 5202.5 s, 2.048M/233M params trainable,
adapter sha 31d36e8d…; VAL forecast batch 4317 origins 11.55 s. Log text không in epoch cuối khi ES dừng
(`lora.py:188-190` break trước log) — epoch đó có trong curve JSON; không sửa (đổi code hash).
loop:tfm = 5 ES fit (calib_seed) + 15 fit fixed-epoch (3 eval seeds × 5 folds), rồi 163 candidates
chỉ fit residual head trên adapter freeze. ETA phụ thuộc best_epoch ES; không giảm workload.
2026-09-12 03:04: 5/5 ES adapter XONG (20:29, 21:55, 23:24, 00:53, ~02:30; mỗi fold 6 epoch, best_epoch 1, ~89 phút).
fixed_epoch_TFM = 1 → khoá adapter dùng hậu tố `_ep1_` (KHÔNG phải `_fixed1_`, đừng grep nhầm khi chờ mốc).
15 fit fixed-epoch mỗi fit ~845–860 s (1 epoch, không ES eval). ES fit: 5171–5310 s/fold, es_mse best theo fold
2.1579 / 0.9071 / 2.5466 / 2.8353 / 0.9929; VAL forecast batch ~11.2–11.8 s cho ~4300 origins mỗi fold.
05:04: 17/20 fit đã chạy (5 es + 5 ep1 seed8587 + 5 ep1 seed8588 + 2 ep1 seed8589), 16 adapter đã lưu.
05:56:37: XONG cả 20 adapter LoRA. rounds = 1 epoch cho mọi fold/seed; ε = 0.3591 pp;
base (TimesFM-LoRA native, S0 = ∅) MedianGain vs E0 = -0.3307 pp → baseline TFM đang KÉM E0, ghi nguyên trong báo cáo.
Vòng add-one 163 candidate: 05:56 → 05:59:32, tất cả KEEP (gain âm nhỏ vẫn trong biên ε), residual head trên
forecast cache nên rất nhanh. F*_raw = 163 KEEP / 0 DROP, MedianGain vs baseline S0 = -3.4233 pp
(keepdrop_tfm.csv). Prune PI 06:00:10 giữ 160/163 → F_pruned (prune_pi_tfm.csv).
06:00 → : CONFIRMATION (§2.1b) fit LoRA ES MỚI cho từng eval seed: `confirm()` ở loop.py:121-132 gọi
run_config(rounds=None) → ES bật, KHÔNG dùng rounds cố định. Đây là phương pháp đã chốt, KHÔNG sửa.
Hệ quả: 15 fit ES mới (3 seed × 5 fold, ~87 phút/fit) ≈ 22 giờ, dự kiến xong ~03:45-04:00 ngày 2026-09-13.
Confirmation F_pruned dùng LẠI chính các adapter đó (khoá adapter theo fold/seed/mode, không theo tập cột),
chỉ fit lại residual head. Tổng adapter khi loop:tfm xong: 20 + 15 = 35 file .pt.
09:04: 22 adapter (2/15 confirmation ES xong). 15:04: 26 adapter = 6/15 confirmation ES xong
(seed8587 đủ 5 fold, seed8588 fold1 xong đang fold2; seed8589 chưa bắt đầu), 821 file lora/residual_heads/*.json,
GPU 85 %, disk 32 GB trống, không lỗi. 21:04: 30 adapter = 10/15 confirmation ES xong (seed8587 + seed8588 đủ 5 fold),
seed8589 (seed cuối) đang fold1. 2026-09-13 01:04: 33 adapter = 13/15 confirmation ES xong (seed8589 lưu 3 fold, đang fold4), còn 2 fit
→ loop:tfm dự kiến xong ~04:00-04:30 cùng ngày. Kích thước output tới lúc này: lora/ 258 MB (adapter .pt ~8.3 MB/cái,
forecast_cache .npy, residual_heads .json), series_preprocess/ 581 MB, CSV/JSON cấp cao 1.4 MB.
2026-09-13 03:55: loop:tfm XONG (118394.6 s = 32.9 h); tfm-final 0.0 s (tái dùng confirmation, không fit lại).
KẾT QUẢ TFM: final = tfm_lora_baseline (System A, feature-free), MedianGain vs E0 = -0.2424 pp; gain từng fold/horizon
đều ÂM (-0.0177 … -1.5130 pp). System B (LoRA + XReg F_win 160 cột) THUA A: MedianGain_B_vs_A = -3.4279 pp,
WinRate 0.0, P10 -119.30, Worst -179.23, ε = 0.3591 → decision = baseline. 15 adapter dùng cho confirmation.
Artifact: wins/tfm.json, wins/tfm_lora_baseline{,_seed0..2}, wins/tfm_lora_xreg{,_seed0..2}, tfm_final.csv,
keepdrop_tfm.csv, prune_tfm.csv, prune_pi_tfm.csv, calib/tfm_base.json, lora/.
03:55:37 → loop:autots_wr: CPU cache 256 cột × 5 fold (1.1-1.3 s/fold), add-one ~3.8 phút/candidate,
123/163 lúc 11:53 → add-one dự kiến xong ~14:20, rồi prune PI + confirmation 3 seed.
GIT: commit stage TFM = 32b1bb3 (2092 file: wins/, lora/, calib/tfm_base.json, 4 CSV tfm, champion_log.csv,
runs/*loop_tfm_*, MEMORY; 0 file experiments/15d). PUSH LẦN 1 HỎNG: LFS Put lên github-cloud S3 bị "i/o timeout"
(lỗi mạng, KHÔNG phải quota/auth; SSH auth OK, remote vẫn là ancestor). Đã set local lfs.concurrenttransfers=3,
activitytimeout=300, dialtimeout=60, tlstimeout=60 rồi retry → PUSH THÀNH CÔNG 12:56 (76 LFS object, 294 MB,
`02fe142..32b1bb3`, xác nhận bằng git log origin/tfm_autots). Băng thông upload thật ~455 KB/s ≈ 1.6 GB/giờ
→ PHẢI push theo từng stage, không dồn vào commit cuối (autots_fits 925 MB + autots_preprocess 2.0 GB + MR sắp tới).
Giữ nguyên 4 config lfs.* local này cho các lần push sau.
BẰNG CHỨNG GPU AutoTS (autots_fits/autots_wr/*.json): prepared_fit.gpu_fit_seconds ≈ 29.5 s tách khỏi
selection_seconds 0.14 s; train_rows 172735, train_columns 153; predict batch 256 origins × 3 steps ~0.016-0.018 s.
autots_fits 925 MB / 1282 file (json+joblib), autots_preprocess 2.0 GB, READY.json preprocessing 36.8 s, 60 base cột/fold.
GPU 0 % ở một lần đọc nvidia-smi là bình thường (xen kẽ CPU chọn cột/metric), không kết luận CPU fallback từ đó.
2026-09-13 16:14:37: loop:autots_wr XONG (44336.1 s = 12.3 h) → loop:autots_mr bắt đầu.
KẾT QUẢ WR: F*_raw 162 KEEP / 1 DROP, MedianGain vs baseline S0 = +0.1193 pp (WR HƠN S0, khác TFM);
prune PI giữ 90/162 cột mới (+21 ext khoá, 72 B0 khoá); F_win = prune (confirmation F_raw vs F_pruned
MedianGain +0.0683 pp, ε = 0.1690) → F_best = 90 cột mới + 21 ext khoá.
Artifact WR: wins/autots_wr.json + autots_wr_seed0..2.npz, keepdrop_autots_wr.csv, prune_autots_wr.csv,
prune_pi_autots_wr.csv, calib/autots_wr_base.json, runs/*autots_wr* (169), autots_fits/autots_wr (1.3 GB, 1740 file),
autots_preprocess/autots_wr (2.0 GB). Chiến lược push: commit kết quả nhỏ TRƯỚC, khối nặng commit/push riêng sau.
16:14:35 → loop:autots_mr: CACHE READY 16:15:37; add-one nhanh hơn WR NHIỀU vì fit MR rẻ thật, không phải CPU fallback:
autots_fits/autots_mr/*.json có gpu_fit_seconds ≈ 1.34 s (WR ≈ 29.5 s), train_rows 172796, 80 cột,
prediction 3.13 s cho 4317 origins (recursive 3 bước), GPU 75 %, VRAM 8.2 GB.
17:06: MR 143/163 candidate, KEEP 30 / DROP 113 (ngược WR vốn gần như KEEP hết).
Dung lượng MR: autots_fits/autots_mr 314 MB, autots_preprocess/autots_mr 2.5 GB. Disk còn 23 GB
(LFS giữ thêm bản sao trong .git/lfs → mỗi GB commit tốn gấp đôi; nếu <10 GB thì xoá ~/.cache/pip 3.5 GB).
17:38:13: loop:autots_mr XONG (5026.5 s = 1.4 h). KẾT QUẢ MR: F*_raw 33 KEEP / 130 DROP,
MedianGain vs baseline S0 = +0.3597 pp (TỐT NHẤT trong ba nhánh); prune PI giữ 21/33 (+8 ext khoá, 72 B0 khoá);
F_win = prune (confirmation MedianGain +0.0137 pp, ε = 0.0050) → F_best = 21 cột mới + 8 ext khoá.
17:38:20: stage CUỐI `autots-search` chạy: bake-off 4 template / 2 nhóm shift ['wr:60','mr'], num_validations=10,
2 frozen set × 5 fold. Sau đó summarize_phase ghi tfm_autots_per_fold_horizon.csv + tfm_autots_summary.csv + PHASE_REPORT.md.
GIT: 1105a38 (kết quả WR nhỏ) đã push. 80554b6 (khối nặng WR 3.3 GB) đang push nền từ 16:20.
QUY TẮC: KHÔNG chạy git add/commit khi tiến trình push nền còn sống (tranh index.lock) — chờ nó xong rồi mới commit stage MR.
BĂNG THÔNG (đo 19:07-19:09 trên /proc/38694/io): push LFS còn sống nhưng chỉ ~64 KB/s (7 MB/120 s), đã truyền ~1.8/3.3 GB
(lúc đầu đo 455 KB/s). Không phải quota/auth. ƯU TIÊN PUSH: (1) kết quả nhỏ + report + checker (vài MB, luôn đẩy được),
(2) khối estimator/cache nhiều GB xếp sau; nếu không kịp thì giữ commit + LFS object local và ghi PUSH_PENDING
đúng từng path kèm số đo, KHÔNG nói đã push/backup ngoài máy.
19:15: đã ghi BẢN NHÁP experiments/tfm_autots/RUN_REPORT.md với toàn bộ evidence đã xác minh (data coverage, provenance
env/checkpoint, stage+runtime thật, kết quả TFM/WR/MR, lỗi đã sửa, hạn chế, bảng trạng thái git). CÒN THIẾU (đánh dấu `CHỜ`):
runtime + kết quả `autots-search`, hai bảng tfm_autots_per_fold_horizon.csv và tfm_autots_summary.csv, mục summary,
và dòng push của các commit MR/summary. Cuối phase chỉ cần điền các mục CHỜ đó rồi commit/push.
21:10 bake-off evidence: 5 fit record có frozen_template (5 fold của đơn vị F_WR_best|wr:60), fit_timing_scope
"native preprocessing plus GPU fit", fit 5.6-56.1 s, predict 0.32 s, prepared_fit=null → KHÔNG có gpu_fit_seconds
tách riêng cho đường native; ghi đúng vậy, không trích số GPU thuần không tồn tại. Không record nào có dấu CPU fallback.
Disk 21:04: còn 23 GB, output 7.0 GB, .git/lfs 3.6 GB, ~/.cache/pip 4.1 GB (chỉ xoá khi disk < 10 GB).
GIT 22:13-22:15: 80554b6 (khối nặng WR 3.4 GB / 919 LFS object) PUSH XONG sau 2 attempt (attempt 1 upload đủ rồi bị
remote đóng kết nối lúc update ref, exit 141; attempt 2 chỉ cần update ref). 8e00a56 (kết quả MR + RUN_REPORT nháp
+ MEMORY) PUSH XONG. Remote hiện ở 8e00a56.
BAKE-OFF 22:15: xong 3/4 đơn vị. F_WR_best|wr:60 (template thắng WindowRegression cả 5 fold),
F_WR_best|mr, F_MR_best|wr:60 (outer VAL @ selection_seed 8587: MedianGain vs E0 = -0.5551 pp). Đơn vị cuối
F_MR_best|mr bắt đầu 21:59:53, dự kiến xong ~00:00. autots_templates đã có 30 file.
CÒN LẠI SAU KHI PHASE DỪNG GHI (commit cuối): autots_fits/* (cả record bake-off mới trong autots_wr),
autots_preprocess/autots_mr 2.5 GB, series_preprocess 581 MB, autots_templates, log.csv, phase_progress.json,
checker_log.jsonl, hai bảng summary, PHASE_REPORT.md, session training.log + exit_code.txt,
RUN_REPORT hoàn chỉnh, CHECKER_FINDINGS mốc cuối. Khoảng 3.4 GB → push nền nhiều giờ, ưu tiên file nhỏ trước.
=== PHASE XONG 2026-09-14 00:26:23 UTC, exit_code=0, phase_progress.status=completed, đủ 6 stage ===
autots-search 24489.9 s (6.8 h). Tổng wall 19:01:35 (09-11) → 00:26:46 (09-14) ≈ 53h25m trên 1× RTX 3090.
Code hash 03d7a4a8e4b34ff25f89cc51f397e898604e5eee0cf42fe96c99532e42c8cec2, config hash 822935ae5e8f.
BAKE-OFF (autots_search.csv, MedianGain vs E0 @ selection_seed 8587, đơn vị pp):
F_WR_best|wr:60 -0.3343 (WindowRegression) | F_WR_best|mr -0.2185 (MultivariateRegression) = FINAL |
F_MR_best|wr:60 -0.5551 | F_MR_best|mr -0.3447. AutoTS-final = F_WR_best|mr, 111 cột ext,
confirmation 3 seed: MedianGain vs E0 = -0.2185 pp, ε = 0.0050; reuse selection predictions cho seed 8587.
SUMMARY (tfm_autots_summary.csv — CHÚ Ý: rmse_gain_vs_e0 và r2_os là PHÂN SỐ, không phải pp):
autots h1/h2/h3: RMSE 54.0196/77.5051/96.4471 vs E0 53.9393/76.1404/92.8937 → gain -0.00101/-0.01322/-0.02801,
R²_OS -0.00203/-0.02684/-0.05788. tfm h1/h2/h3: RMSE 54.0865/76.5147/93.5740 → gain -0.00223/-0.00404/-0.00613,
R²_OS -0.00448/-0.00811/-0.01233. KẾT LUẬN THẬT: KHÔNG family nào thắng E0 ở giá thô trên VAL.
CHECKER MỐC CUỐI XONG: PASS C1-C6 (stages/split/seeds, 35 adapter + 845 residual head + 1740 fit record + 30 native
record + 3 READY.json không còn .building, R² OS đúng công thức, causality FIT/ES/VAL, 0 record device cpu,
0 file experiments/15d trong 02fe142..HEAD). ERROR/WARN đã SỬA HẾT trong RUN_REPORT + CHECKER_FINDINGS:
E1 AutoTS seed dispersion = 0 do template ghim random_state 8587 (npz 3 seed trùng sha256) → không gọi là mean 3 seed;
E2 summary là mean theo fold, không pooled, đơn vị phân số; E3 thay số mẫu đơn lẻ bằng dải thật (WR gpu_fit 29.09-71.05 s
/153-315 cột, MR 0.585-1.403 s/80-116 cột, bake-off 30 record fit 4.24-74.76 s, ES fit 5079.4-5313.7 s, fixed 836.6-865.0 s);
E4 bỏ stage "summary" không tồn tại; E6 KHÔNG có MAE cho đại diện cuối và KHÔNG có latency; E7 Worst -179.23 do fold4 bung;
E9 utilisation GPU không kiểm được từ artifact (chỉ device cuda + gpu_fit_seconds), nvidia-smi chỉ là quan sát session.
02:24: dde5539 (1909 file: autots_fits/*, autots_preprocess/autots_mr, series_preprocess) PUSH XONG —
937 LFS object ~4.0 GB, hai attempt (attempt 1 upload đủ rồi remote đóng kết nối lúc update ref, exit 141;
attempt 2 update ref). origin/tfm_autots = dde5539. KHÔNG còn path nào PUSH_PENDING. Disk 19 GB.
CÒN LẠI: commit/push cuối (RUN_REPORT + CHECKER_FINDINGS mốc 2 + MEMORY) rồi xác minh origin chứa commit đó.
I1 (NCCL) CHỐT: process map cả hai bản — torch nvidia/nccl 2.28.9 và hệ thống 2.31.2 — lib_lightgbm.so đã nạp;
single-GPU nên LightGBM không init NCCL collectives. Disk: output 3.8 GB, còn 29 GB.
Khi loop:tfm xong: commit/push stage (keepdrop_tfm.csv, prune_pi_tfm.csv, wins/tfm*, lora/, log CSV) bằng path tường minh,
cân nhắc kích thước LFS trước khi add cache series_preprocess; rồi theo tfm-final và hai loop AutoTS.
Kế tiếp: vòng add-one 163 candidate (log dạng `[tfm] NNN <tên> Median … → KEEP/DROP`,
`cli.py:714`), chỉ fit residual head. Sau đó tfm-final → loop autots_wr → loop autots_mr → autots-search → summary.
Đã push 02fe142 (setup provenance + lock-s0 + CHECKER_FINDINGS mốc 1).

Exact next step: theo dõi tmux/log/phase_progress.json. Process chết → giữ traceback + run dir, sửa nguyên nhân,
chạy lại launcher (tự --resume). Sửa src/p0 hoặc src_OB/gpu.py làm đổi code hash → phase guard từ chối output cũ:
cần config recovery chỉ đổi experiments_dir, ghi provenance. Stage xong → commit/push output + session logs
(không add -A, không file đang ghi dở). Cuối phase: checker cuối → experiments/tfm_autots/RUN_REPORT.md,
CHECKER_FINDINGS.md → `git push origin tfm_autots` (pushurl SSH đã auth tson295).
Không chạy bootstrap/gpu-probe/check-data riêng/test/smoke hoặc pipeline OB. Không giảm workload.
Ghi chú OB trước đây đã archive ở docs/archive/ob_before_tfm_vast_2026-09-11/.
