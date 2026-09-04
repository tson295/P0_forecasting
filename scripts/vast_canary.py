"""Canary PACKAGE THẬT trước full run (chạy trên Vast, sau `scripts/vast_bootstrap.sh`).

Unit test local dùng stub nên KHÔNG chứng minh được timesfm/autots thật khớp adapter. Script này gọi đúng
đường code của harness (`p0.models_tfm`, `p0.models_autots`, `p0.autots_search`) với package thật, trên
**data tổng hợp** (không đụng data thật, không phải training của experiment), và đo thời gian để ra ETA.

Mọi lỗi → in traceback đầy đủ + exit non-zero. KHÔNG tự đổi methodology, KHÔNG fallback CPU.

    PYTHONPATH=src:. python scripts/vast_canary.py --config configs/p0_15d.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from p0.config import HORIZONS, RunConfig  # noqa: E402
from p0.harness import ColSet, Store, _standardize_fit, run_config  # noqa: E402
# Canary dùng `tfm_zero_shot` (lớp tham chiếu): cùng đường forecast/covariate/xreg GPU với TimesFMLoRAModel, không cần train LoRA.
from p0.models import SeriesBatch, make_model  # noqa: E402
from p0.split import make_folds  # noqa: E402
from p0.synthetic import make_hf, make_lf  # noqa: E402

RESULTS: list[dict] = []


def check(name: str, fn, note: str = ""):
    t0 = time.perf_counter()
    try:
        extra = fn() or ""
        dt = time.perf_counter() - t0
        RESULTS.append({"check": name, "ok": True, "sec": round(dt, 2), "note": str(extra) or note})
        print(f"  OK   {name:<44s} {dt:7.2f}s  {extra}", flush=True)
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        RESULTS.append({"check": name, "ok": False, "sec": round(dt, 2), "note": f"{type(e).__name__}: {e}"})
        print(f"  FAIL {name:<44s} {dt:7.2f}s  {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p0_15d.json")
    ap.add_argument("--origins", type=int, default=24, help="số origin cho canary (nhỏ, chỉ để kiểm API + đo tốc độ)")
    args = ap.parse_args()
    cfg = RunConfig.load(ROOT / args.config)
    lgb_dev = cfg.model_params("lgbm").get("device_type", "gpu")
    xgb_dev = cfg.model_params("xgb").get("device", "cuda")
    print(f"backend đã resolve: LightGBM device_type={lgb_dev} · XGBoost device={xgb_dev} · CatBoost GPU · torch CUDA\n")

    # ---------------------------------------------------------------- data tổng hợp (KHÔNG đụng data thật)
    hf = make_hf(n_days=6.0, seed=7)
    store = Store(hf, make_lf(hf))
    start = str(hf["datetime"].iloc[0])[:10]
    import pandas as pd

    d0 = pd.Timestamp(start)
    folds = make_folds(store.first_origin_ts, [(d0 + pd.Timedelta(days=4)).strftime("%Y-%m-%d")], purge_minutes=60)
    val_idx = folds[0].val.origins(store.ts, store.eligible)[: args.origins]
    b0 = ColSet(store.b0_names[:12])
    ext2 = ("ret_60", "bb_pctb_20")

    # ---------------------------------------------------------------- 1. tree + LSTM trên GPU thật
    print("== 1. GPU fit thật: LightGBM / XGBoost / CatBoost / LSTM ==")

    def _tree(name):
        m = make_model(name, cfg.model_params(name))
        r = run_config(store, m, ColSet(store.b0_names[:20]), folds, rounds=None, seed=cfg.calib_seed, keep_states=False)
        assert np.isfinite(r.rmse).all() and (r.rmse > 0).all(), r.rmse
        return f"RMSE={np.round(r.rmse[0], 2).tolist()} best_iters={r.best_iters[0].tolist()}"

    for name in ("lgbm", "xgb", "cat", "xgbrf"):
        check(f"{name} fit GPU (1 fold)", lambda n=name: _tree(n))

    def _lstm():
        m = make_model("lstm", {**cfg.model_params("lstm"), "context": 64, "hidden": 8, "max_epochs": 1, "batch_size": 256})
        r = run_config(store, m, ColSet(store.b0_names[:22], ("ret_60",)), folds, rounds=(1, 1, 1), seed=cfg.calib_seed,
                       keep_states=False)
        assert np.isfinite(r.rmse).all()
        return f"RMSE={np.round(r.rmse[0], 2).tolist()} (train device {m.train_device})"

    check("LSTM forward+train CUDA", _lstm)

    # ---------------------------------------------------------------- 2. TimesFM package thật
    print("\n== 2. TimesFM (package thật, checkpoint thật) ==")
    import timesfm  # noqa: F401

    check("timesfm import + version", lambda: f"timesfm {timesfm.__version__ if hasattr(timesfm, '__version__') else '2.0.2'}")
    tfm_native = make_model("tfm_zero_shot", cfg.model_params("tfm"))
    r1 = store.r1
    cov = _standardize_fit(np.column_stack([store.ext_column(c) for c in ext2]).astype(np.float32),
                           folds[0].fit.origins(store.ts, store.eligible))

    state = {}

    def _tfm_load():
        m = tfm_native._model(with_covariates=False)  # load checkpoint + compile (per_core_batch_size = batch_size)
        state["m"] = m
        return f"checkpoint {tfm_native.repo_id}@{tfm_native.revision[:12]}"

    check("TimesFM load checkpoint + compile (native)", _tfm_load)

    def _tfm_native():
        seq = SeriesBatch(store.ts, r1, val_idx)
        t0 = time.perf_counter()
        yhat = tfm_native.predict_series(seq)
        state["native_ms"] = 1000 * (time.perf_counter() - t0) / max(1, len(val_idx))
        assert yhat.shape == (len(val_idx), len(HORIZONS)), yhat.shape
        assert np.isfinite(yhat).all(), "native forecast có NaN/inf"
        assert np.abs(yhat).max() < 0.5, f"|ŷ| bất thường: {np.abs(yhat).max()}"
        return f"shape={yhat.shape} |ŷ|max={np.abs(yhat).max():.2e} {state['native_ms']:.1f} ms/origin"

    check("TimesFM native forecast (chỉ r1)", _tfm_native)

    def _tfm_head():
        # head phải là MEAN (quantile[...,0]), không phải q50 (point_forecast)
        ctxs = tfm_native.contexts(SeriesBatch(store.ts, r1, val_idx[:4]))
        res = state["m"].forecast(horizon=len(HORIZONS), inputs=list(ctxs))
        assert isinstance(res, (tuple, list)) and len(res) >= 2, f"forecast() trả {type(res)}"
        point, quant = np.asarray(res[0]), np.asarray(res[1])
        assert quant.ndim == 3 and quant.shape[2] >= 10, f"quantile shape {quant.shape}"
        mean_head = tfm_native.head(res)
        assert np.allclose(mean_head[:, : len(HORIZONS)], quant[:, : len(HORIZONS), 0]), "head() không lấy kênh mean"
        d = float(np.max(np.abs(point[:, : len(HORIZONS)] - quant[:, : len(HORIZONS), 0])))
        return f"point(q50) vs mean lệch tối đa {d:.3e} (head dùng mean)"

    check("TimesFM mean-head = quantile[...,0]", _tfm_head)

    def _tfm_cov():
        assert tfm_native.forecast_config_kwargs(True)["per_core_batch_size"] == 1
        m_cov = make_model("tfm_zero_shot", cfg.model_params("tfm"))
        seq = SeriesBatch(store.ts, r1, val_idx[:8], cov, ext2)
        t0 = time.perf_counter()
        yhat = m_cov.predict_series(seq)
        state["cov_ms"] = 1000 * (time.perf_counter() - t0) / 8
        assert yhat.shape == (8, len(HORIZONS)) and np.isfinite(yhat).all()
        return f"{state['cov_ms']:.0f} ms/origin (batch=1, {len(ext2)} covariate)"

    check("TimesFM covariate forecast (per_core_batch_size=1)", _tfm_cov)

    def _tfm_causal():
        """Cắt chuỗi sau origin cuối → prediction phải GIỐNG HỆT (chỉ dùng τ ≤ t)."""
        idx = val_idx[:6]
        a = make_model("tfm_zero_shot", cfg.model_params("tfm")).predict_series(SeriesBatch(store.ts, r1, idx, cov, ext2))
        r1b, covb = r1.copy(), cov.copy()
        cut = int(idx[-1]) + 1
        r1b[cut:] = 0.0
        covb[cut:] = 0.0
        b = make_model("tfm_zero_shot", cfg.model_params("tfm")).predict_series(SeriesBatch(store.ts, r1b, idx, covb, ext2))
        assert np.allclose(a, b, atol=1e-6), f"prediction đổi khi cắt dữ liệu sau t (max Δ={np.abs(a - b).max():.2e})"
        w = make_model("tfm_zero_shot", cfg.model_params("tfm")).covariate_window(
            SeriesBatch(store.ts, r1, idx, cov, ext2), int(idx[0]), 0)
        L = 512
        assert len(w) == L + len(HORIZONS)
        assert np.allclose(w[:L], cov[int(idx[0]) - L:int(idx[0]), 0]), "covariate không dịch 1 bar"
        assert np.allclose(w[L:], cov[int(idx[0]), 0]), "3 bước tương lai không giữ f(t)"
        return "cắt-chuỗi bit-identical; covariate dịch đúng 1 bar; 3 bước tương lai = f(t)"

    check("TimesFM causal (τ ≤ t) + shift 1 bar", _tfm_causal)

    # ---------------------------------------------------------------- 3. AutoTS package thật
    print("\n== 3. AutoTS (package thật) ==")
    import autots  # noqa: F401

    check("autots import + version", lambda: f"autots {autots.__version__}")
    small = ColSet(store.b0_names[:6], ("ret_60",))

    def _probe(kind):
        from p0 import cli

        m = cli._autots_probe_model(cfg, "wr:60" if kind == "wr" else "mr", allow_cpu=False)
        dev = m.regression_model["model_params"].get("device_type") or m.regression_model["model_params"].get("device")
        t0 = time.perf_counter()
        r = run_config(store, m, small, folds, rounds=None, seed=cfg.sel_seed, keep_states=True)
        n = len(r.states[0].idx_val)
        state[f"{kind}_ms"] = 1000 * (time.perf_counter() - t0) / max(1, n)
        assert np.isfinite(r.rmse).all() and (r.rmse > 0).all()
        return f"{m.regression_model['model']}({dev}) RMSE={np.round(r.rmse[0], 2).tolist()} {state[f'{kind}_ms']:.1f} ms/origin"

    check("AutoTS-WR probe (LightGBM GPU, rolling)", lambda: _probe("wr"))
    check("AutoTS-MR probe (XGBoost cuda, rolling)", lambda: _probe("mr"))

    def _regressor_used():
        """future_regressor phải thật sự ảnh hưởng prediction (nếu bị bỏ qua thì đổi regressor không đổi kết quả)."""
        from p0 import cli

        m = cli._autots_probe_model(cfg, "mr", allow_cpu=False)
        idx_fit = folds[0].fit.origins(store.ts, store.eligible)
        cov_a = _standardize_fit(np.column_stack([store.ext_column("ret_60")]).astype(np.float32), idx_fit)
        seq_fit = SeriesBatch(store.ts, r1, idx_fit, cov_a, ("ret_60",))
        seq_val = SeriesBatch(store.ts, r1, val_idx[:6], cov_a, ("ret_60",))
        a = m.fit_predict(seq_fit, None, None, None, seq_val, None, cfg.sel_seed).pred_z
        cov_b = cov_a.copy() * -3.0
        m2 = cli._autots_probe_model(cfg, "mr", allow_cpu=False)
        b = m2.fit_predict(SeriesBatch(store.ts, r1, idx_fit, cov_b, ("ret_60",)), None, None, None,
                           SeriesBatch(store.ts, r1, val_idx[:6], cov_b, ("ret_60",)), None, cfg.sel_seed).pred_z
        assert not np.allclose(a, b), "đổi regressor không đổi prediction → AutoTS đang BỎ QUA future_regressor"
        return f"đổi regressor → prediction đổi (max Δ={np.abs(a - b).max():.2e})"

    check("AutoTS thật sự dùng future_regressor", _regressor_used)

    def _bakeoff():
        from p0 import cli
        from p0.autots_search import template_frame

        groups, nv = cli.autots_search_cfg(cfg)
        reg = cli.autots_regressors(cfg)
        for g, specs in groups.items():
            tmpl = template_frame(specs, seed=cfg.sel_seed, regressors=reg)
            for _, row in tmpl.iterrows():  # mọi dòng template phải GPU-safe
                p = json.loads(row["ModelParameters"])
                rm = p["regression_model"]["model_params"]
                dev = rm.get("device_type") or rm.get("device")
                assert dev in (lgb_dev, xgb_dev), f"template {row['Model']} dùng device {dev!r} (không phải backend đã resolve)"
                assert str(p.get("regression_type", "")).lower() == "user"
                assert p.get("datepart_method") in (None, "None") and not p.get("holiday", False)
        g0 = next(iter(groups))
        t0 = time.perf_counter()
        name, params, _all = cli.autots_bakeoff_fold(cfg, store, folds[0], small, g0, groups[g0], min(nv, 2), False)
        state["bakeoff_sec"] = time.perf_counter() - t0
        n_fit = len(groups[g0]) * (min(nv, 2) + 1)
        state["fit_sec"] = state["bakeoff_sec"] / max(1, n_fit)
        assert str(params.get("regression_type", "")).lower() == "user"
        return f"best={name}, {n_fit} fit trong {state['bakeoff_sec']:.0f}s → {state['fit_sec']:.1f}s/fit"

    check("AutoTS framework bake-off (max_generations=0)", _bakeoff)

    # ---------------------------------------------------------------- 4. ETA
    print("\n== 4. ETA (từ thời gian đo được, 5 fold × 1.437 origin = 7.185 origin/run) ==")
    N_ORIGIN, N_CAND = 7185, 164  # base + 163 candidate C_short (vòng expanded-data; VAL 3 ngày × 5 fold = 21.600 origin/run trên data thật)
    eta = {}
    if "native_ms" in state:
        eta["tfm native (1 run)"] = N_ORIGIN * state["native_ms"] / 3.6e6
    if "cov_ms" in state:
        eta["tfm XReg add-one (164 run)"] = N_CAND * N_ORIGIN * state["cov_ms"] / 3.6e6
    for k, lab in (("wr_ms", "autots_wr probe (40 run)"), ("mr_ms", "autots_mr probe (40 run)")):
        if k in state:
            eta[lab] = N_CAND * N_ORIGIN * state[k] / 3.6e6
    if "fit_sec" in state:
        groups, nv = __import__("p0.cli", fromlist=["cli"]).autots_search_cfg(cfg)
        n_fit = sum(len(v) for v in groups.values()) * (nv + 1) * 5 * 2  # × 5 fold × ≤ 2 frozen set
        eta["autots-search (bake-off)"] = n_fit * state["fit_sec"] / 3600
    for k, v in eta.items():
        print(f"  {k:<34s} ≈ {v:6.1f} h")
    print("  (tree/LSTM: đo trực tiếp ở phase tương ứng; ETA cập nhật lại sau mỗi phase bằng thời gian thật)")

    out = ROOT / "experiments" / "canary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": {"lightgbm": lgb_dev, "xgboost": xgb_dev}, "checks": RESULTS,
                               "timing_ms_per_origin": {k: round(v, 2) for k, v in state.items() if k.endswith("_ms")},
                               "eta_hours": {k: round(v, 2) for k, v in eta.items()}}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    bad = [r["check"] for r in RESULTS if not r["ok"]]
    print(f"\n→ {out}")
    if bad:
        print(f"CANARY FAIL ({len(bad)}): {bad}\nKHÔNG được chạy experiment. Báo user kèm traceback ở trên.")
        return 1
    print(f"CANARY PASS ({len(RESULTS)} mục).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
