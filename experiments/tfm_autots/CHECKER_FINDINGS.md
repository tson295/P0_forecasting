# Checker findings — tfm_autots (BTC 1m two years)

Checker = read-only subagent running the instructions of `.claude/agents/checker.md` (the project agent type was not
registered in this Claude session because the session started outside the repo, so a general-purpose subagent was given
the same file, rules and tool limits). It did not write files, install, run tests/probes or import model libraries.
The main session applied every resolution below.

## Milestone 1 — after setup (2026-09-11 ~19:05 UTC, run `run_20260911T190135Z_Jmb166`)

Result: no ERROR. GPU fits were UNVERIFIED at that time (only `torch.cuda.is_available()` at gate).

| ID | Level | Evidence | Status / resolution |
|---|---|---|---|
| P1 | PASS | `configs/tfm_autots.json` sha256 == session `config.json`; folds/seeds/TimesFM/LoRA/residual/AutoTS search/model_order/defer_champion/require_gpu as locked | — |
| P2 | PASS | sha256 + bytes of both CSVs match `data_checksums_2y.json` / `BTC_5m_2y.derivation.json`; 1m 1,051,201 rows 2024-09-03 16:29 → 2026-09-03 16:29; phase load uses `load_store(verify=True)` (`cli.py:184-186`) | — |
| P3 | PASS | 15d S0 sources are real files (not LFS pointers); WR 72 B0* + 21 ext, MR 72 + 8, TimesFM S0 = ∅; 163 candidates per model; `checker_log.jsonl` S0_LOCK PASS | — |
| P4 | PASS | no jax/jaxlib/flax; autots 1.0.4, timesfm 2.0.2, torch 2.11.0+cu128, xgboost 3.2.0 GPU wheel; `lib_lightgbm.so` sm_86 CUDA code, NEEDED `libnccl.so.2`, build flags in `setup_logs/lgbm_build.log` | — |
| P5 | PASS | launcher PID 4320 → run.py PID 4339 with `P0_TFM_AUTOTS_VAST=1`, `CUDA_VISIBLE_DEVICES=0`, `HF_HOME=/home/ubuntu/.cache/huggingface`; HEAD d9201e1, empty `code_changes.patch` | — |
| P6 | PASS | HF snapshot `1d952420…` == `REVISION` pin `src/p0/models_tfm.py:32`, no incomplete blobs | — |
| W1 | WARN | 71 files under `experiments/15d` (27 `wins/*.npz`, 44 PNG) show as modified; blob hash == index hash; cause: `.gitattributes` LFS rules over files committed as plain blobs | Never staged. Commits use explicit paths only (`experiments/tfm_autots/…`, `experiments/tfm_autots_sessions/…`, `.claude/MEMORY.md`); no `git add -A/-u`, no `commit -a` |
| W2 | WARN | build/pip logs only in the session scratchpad; `environment.json` lacks HF_HOME/revision/LightGBM flags | Resolved: logs copied to `experiments/tfm_autots_sessions/setup_logs/`; `SETUP_ENV.md` records HF_HOME, revision, build flags, NCCL note |
| I1 | INFO | `lib_lightgbm.so` has no RPATH; runtime NCCL copy depends on load order | To be read from `/proc/<pid>/maps` in AutoTS stages (see milestone 3) |
| I2 | INFO | pandas `divide by zero encountered in log` in training.log; likely `Baseline_LGBM.py:368` log(rv5/rv60) when rv5 = 0, inf→NaN at `features_short.py:311` | Existing feature behaviour; `Baseline_LGBM.py` is frozen, not edited |
| I3 | INFO | `phase_config.json` has default `es_hours: 23`, only used by legacy `make_folds`; rolling_spread uses `es_days=5` | No effect |
| I4 | INFO | at check time: lock-s0 done (37 s), `loop:tfm` in CPU cache stage, GPU idle | Superseded: at 19:04:58 the first LoRA fit was running (GPU 91 %, 5.2 GB, 209 W); completion evidence in later milestones |
| I5 | INFO | tmux name / `pip check` / cache disk sizing not verified by checker | Main session: tmux `p0_tfm_autots` listed by `tmux ls`; `pip check` output "No broken requirements found." in session; disk 33 GB free at 19:09 |

## Milestone 2 — cuối phase (2026-09-14, sau khi run kết thúc exit 0)

Checker đọc artifact cuối, read-only, không train/infer/test và không chạy lệnh git ghi trạng thái.

### PASS có bằng chứng

| ID | Evidence |
|---|---|
| C1 | `phase_progress.json` đủ 6 stage + `stage_seconds`, khớp từng mốc trong `training.log`: lock-s0 19:01:39→19:02:16 (37,0 s), loop:tfm →09-13 03:55:31 (118.394,6 s), tfm-final 0,006 s, WR →16:14:27 (44.336,1 s), MR →17:38:13 (5.026,5 s), autots-search →09-14 00:26:23 (24.489,9 s); `exit_code.txt` = 0 |
| C2 | 163 candidate/model sau S0 collision handling (`checker_log.jsonl` CANDIDATE_M, `s0/candidates_*.json`, 163 dòng mỗi `keepdrop_*.csv`); 5 fold với FIT 172.797 bar, inner ES 7.197, residual suffix 7.197, purge 3.600 s, VAL 4.300–4.317 origin; không có artifact TEST; seeds 8586/8587-8589/8587 đúng ở `lora/*.json`, `autots_fits/*`, `log.csv`; `runs/` = 510 = 163×3 + 21 confirmation |
| C3 | 35 `lora/*.pt` = 5 ES@8586 + 15 `_ep1_` + 15 ES confirmation; 845 residual_heads, 885 runtime, 55 forecast_cache; 1.740 fit record đường search + 30 record native; 3 `READY.json` (key gồm code/data/dependencies/recipe/schema/columns/fold_indices/seeds; 256/243/163 cột), **không còn `.building.*`**; 40 template JSON |
| C4 | Cả 30 ô R² OS đúng `1 − mean(MSE theo seed)/E0²` (`phase_tfm_autots.py:120`), không phải bình phương mean-seed RMSE; hai bảng suy ra bit-đối-bit từ `wins/tfm.json` và `wins/autots.json` |
| C5 | Chuỗi `lora_fit → inner_es → residual_fit` nối nhau với purge 3.600 s và kết thúc đúng cuối FIT; outer ES chỉ vào refit của bake-off native (179.937 bar = FIT+ES); VAL không bao giờ được fit; không jax, không `forecast_with_covariates`, `wins/tfm.json:"xreg": null`; 0 record `device: cpu`/fallback; log không có traceback/OOM |
| C6 | `origin/tfm_autots` = `cbb9cb6`; mọi path output trên đĩa đều có trong HEAD; **0 file `experiments/15d`** trong `02fe142..HEAD`; không venv/secret/checkpoint pretrained; `.pt/.npz/.npy/.joblib` đều là LFS pointer |

### Findings và xử lý

| ID | Mức | Nội dung | Xử lý |
|---|---|---|---|
| E1 | ERROR | `wins/autots.json` có `seed_rmse` giống hệt ở 3 eval seed và `autots_seed{0,1,2}.npz` trùng SHA256; nguyên nhân: template đông lạnh ghim `random_state: 8587` và MR không bootstrap, nên 10 refit confirmation tái tạo đúng cùng một fit | RUN_REPORT §4.5 nêu rõ độ phân tán seed của AutoTS bằng 0 do cấu trúc, không gọi là trung bình 3 seed độc lập; TFM và WR khác biệt thật theo seed |
| E2 | WARN | `tfm_autots_summary.csv` là groupby-mean của các ô per-fold (`phase_tfm_autots.py:126-128`), không recompute; thiếu cột `aggregation`/`e0_status` | §4.5 ghi "trung bình không trọng số theo fold", không gọi pooled, và nói rõ đơn vị là phân số |
| E3 | ERROR | Bản nháp trích mẫu đơn lẻ: WR "29,5 s/153 cột", MR "1,34 s/80 cột", batch "0,016–0,018 s", bake-off "5 record", fixed-epoch/ES thiếu dải | Thay bằng dải thật: WR 29,09–71,05 s (trung vị 48,4) và 153–315 cột, batch 0,0123–0,0566 s; MR 0,585–1,403 s, 80–116 cột, batch 0,117–2,95 s; bake-off 30 record, fit 4,24–74,76 s, predict 0,129–4,009 s; ES fit 5.079,4–5.313,7 s (20 fit), fixed 836,6–865,0 s |
| E4 | ERROR | §3 có dòng stage "summary" không tồn tại; `autots-search` chưa điền; §7 nói `80554b6` đang push | §3 bỏ dòng đó và ghi `summarize_phase` chạy trong `autots-search` (24.489,9 s); §7 cập nhật đúng trạng thái push |
| E5 | must-add | Thiếu bảng bake-off và kết luận đối chiếu E0 | §4.4 thêm bảng 4 đơn vị; §4.5 nêu thẳng không family nào thắng E0 |
| E6 | WARN | Không có MAE cho đại diện cuối; không có latency nào | §6 liệt kê rõ là chưa xuất, kèm cảnh báo batch ≠ p95 single-request |
| E7 | INFO | Worst −179,23 của hệ B do fold4 bung (RMSE 128,6/183,0 vs E0 45,9/65,9) | §4.1 mô tả đúng là một fold bung, không phải suy giảm đều |
| E8 | WARN | `CHECKER_FINDINGS.md` mới có mốc 1 | Chính mục này |
| E9 | INFO | 30 record native có `prepared_fit: null`; utilisation GPU không kiểm được từ artifact | §4.4 giữ nguyên cách diễn đạt; §6 tách rõ quan sát `nvidia-smi` của session với bằng chứng artifact |

