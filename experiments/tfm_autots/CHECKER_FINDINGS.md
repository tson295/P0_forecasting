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
