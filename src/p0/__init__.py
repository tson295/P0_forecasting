"""P0_forecasting harness — implement docs/RESEARCH_PLAN.md (rev 10, 2026-09-03: vòng expanded-data).

Module map (§8 plan):
- config.py        : RunConfig (JSON: split rolling / val_days, checksums, prev_run_dir, fold_workers, short_candidates), config_hash
- data.py          : load CSV, adapter lowercase→B0 uppercase, kiểm tra §1.1, checksum, as-of join LF 5'
- split.py         : fold §1.2 (15 ngày) + RollingSpec / make_rolling_from_end §1.5 (data đầy đủ, neo cuối data), purge 60', TEST
- features_ext.py  : 39 candidate §2.3 (vòng 15 ngày — lịch sử; cột KEEP nằm trong S0_m)
- features_short.py: C_short §2.3b — candidate ngắn hạn ≤ 15' (vòng expanded-data)
- s0.py            : S0_m khoá từ artifact vòng trước, collision audit bằng số, Candidate_m, lock/load
- metrics.py       : metric trên giá §0, Gain 15 ô, tóm tắt, gộp 3 seed (mean RMSE từng ô)
- models.py        : adapter LightGBM (B0), XGBoost, XGB-RF, CatBoost + SeriesBatch; models_lstm.py (LSTM),
                     models_tfm.py (TimesFM 2.5 zero-shot tham chiếu + TimesFMLoRAModel: LoRA per fold → freeze → XReg),
                     lora.py (LoRA tự chứa + train loop), models_autots.py (WindowRegression / MultivariateRegression), autots_search.py
- harness.py       : Store (B0 matrix + ext/short), ColSet (b0/ext/locked), run_config (+ fold-parallel), calibrate, seed_noise (ε_m)
- fold_parallel.py : 5 fold chạy song song ở process riêng cho mọi model (§0b.6) — kết quả y hệt tuần tự
- filter_b0.py     : §1.4 PI / SA / MI → cờ ≥ 2/3 horizon → R1–R4 → kiểm chứng → B0* (đã xong ở vòng 15 ngày)
- loop.py          : §2.1 add-one, prune PI (chỉ cột mới), confirmation 3 seed → win_m; §3 champion log, ensemble
- latency.py       : §7.4 pass đo riêng, batch 1, p95/p99/max
- plots.py         : định nghĩa figure §7.3; visualize.py : dựng lại mọi figure từ artifact (hậu kỳ, không train)
- cli.py           : python run.py <step> --config configs/p0_full.json (check-data, lock-s0, calibrate, filter-b0, loop, tfm-final,
                     autots-search, ensemble, final, visualize, smoke-e2e)
"""
