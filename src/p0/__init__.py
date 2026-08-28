"""P0_forecasting harness — implement docs/RESEARCH_PLAN.md (rev 9b).

Module map (§8 plan):
- config.py        : RunConfig (JSON), config_hash
- data.py          : load CSV, adapter lowercase→B0 uppercase, kiểm tra §1.1, checksum, as-of join LF 5'
- split.py         : fold §1.2 (half-open, t + 3' < T_end), purge 60', TEST
- features_ext.py  : 39 candidate §2.3 (causal, lookback ≤ 1440 trừ #35)
- metrics.py       : metric trên giá §0, Gain 15 ô, tóm tắt, gộp 3 seed (mean RMSE từng ô)
- models.py        : adapter LightGBM (B0), XGBoost, XGB-RF, CatBoost + SeriesBatch; models_lstm.py (LSTM),
                     models_tfm.py (TimesFM 2.5), models_autots.py (WindowRegression / MultivariateRegression)
- harness.py       : Store (B0 matrix + ext), run_config, calibrate (15fixed_m), seed_noise (ε_m)
- filter_b0.py     : §1.4 PI / SA / MI → cờ ≥ 2/3 horizon → R1–R4 → kiểm chứng → B0*
- loop.py          : §2.1 add-one, prune PI, confirmation 3 seed → win_m; §3 champion log, ensemble
- latency.py       : §7.4 pass đo riêng, batch 1, p95/p99/max
- plots.py         : §7.3 Fig P (forecast path t → t+3 của một origin), Fig HM, Final heatmap + Fig P mọi model
- cli.py           : python run.py <step> --config configs/p0_15d.json
"""
