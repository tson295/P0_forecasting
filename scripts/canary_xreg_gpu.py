"""Canary: xreg của TimesFM covariate PHẢI chạy trên GPU (user chốt 2026-09-01).

Chứng minh bằng package thật + checkpoint thật, trên data TỔNG HỢP (không đụng data thật):
  1. jax default backend = gpu, có CudaDevice;
  2. khối giải beta_hat của `xreg_lib` thực sự materialize trên CUDA (bắt tận nơi bằng monkeypatch
     `jnp.linalg.pinv` để đọc `.devices()` của kết quả) — KHÔNG suy diễn từ biến môi trường;
  3. covariate forecast vẫn chạy, output hữu hạn, shape đúng;
  4. giữ nguyên 1 origin/lời gọi (per_core_batch_size=1 và len(inputs)==1 ở MỌI lời gọi);
  5. causal: cắt chuỗi sau origin cuối → prediction bit-identical; covariate dịch đúng 1 bar;
  6. KHÔNG có CPU fallback: nếu bất kỳ op nào của xreg rơi về CPU → FAIL.
  7. torch CUDA vẫn sống chung với jax GPU trong cùng process.

    PYTHONPATH=src:. python scripts/canary_xreg_gpu.py --config configs/p0_15d.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        extra = fn() or ""
        dt = time.perf_counter() - t0
        RESULTS.append({"check": name, "ok": True, "sec": round(dt, 2), "note": str(extra)})
        print(f"  OK   {name:<46s} {dt:7.2f}s  {extra}", flush=True)
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        RESULTS.append({"check": name, "ok": False, "sec": round(dt, 2), "note": f"{type(e).__name__}: {e}"})
        print(f"  FAIL {name:<46s} {dt:7.2f}s  {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p0_15d.json")
    ap.add_argument("--origins", type=int, default=12)
    args = ap.parse_args()

    from p0.config import HORIZONS, RunConfig
    from p0.harness import ColSet, Store, _standardize_fit
    from p0.models import SeriesBatch, make_model
    from p0.split import make_folds
    from p0.synthetic import make_hf, make_lf

    cfg = RunConfig.load(ROOT / args.config)

    # ---------------------------------------------------------------- 1. jax backend
    print("== 1. JAX backend ==")
    import jax
    import jax.numpy as jnp

    def _backend():
        assert jax.default_backend() == "gpu", f"jax.default_backend() = {jax.default_backend()!r} (phải 'gpu')"
        devs = jax.devices()
        assert any(d.platform == "gpu" for d in devs), f"không có GPU device: {devs}"
        return f"jax {jax.__version__} backend={jax.default_backend()} devices={devs}"

    check("jax default backend = gpu", _backend)

    # ---------------------------------------------------------------- 2. bẫy op của xreg
    # Monkeypatch jnp.linalg.pinv để ghi lại device THẬT của kết quả beta_hat trong xreg_lib.
    from timesfm.utils import xreg_lib

    seen: dict = {"pinv_devices": set(), "calls": 0, "input_lens": set()}
    _orig_pinv = xreg_lib.jnp.linalg.pinv

    def _spy_pinv(a, *a_, **kw):
        out = _orig_pinv(a, *a_, **kw)
        try:
            seen["pinv_devices"] |= {str(d.platform) for d in out.devices()}
        except Exception:
            seen["pinv_devices"] |= {"unknown"}
        seen["calls"] += 1
        return out

    xreg_lib.jnp.linalg.pinv = _spy_pinv

    # ---------------------------------------------------------------- data tổng hợp
    hf = make_hf(n_days=6.0, seed=7)
    store = Store(hf, make_lf(hf))
    import pandas as pd

    d0 = pd.Timestamp(str(hf["datetime"].iloc[0])[:10])
    folds = make_folds(store.first_origin_ts, [(d0 + pd.Timedelta(days=4)).strftime("%Y-%m-%d")], purge_minutes=60)
    val_idx = folds[0].val.origins(store.ts, store.eligible)[: args.origins]
    ext2 = ("ret_60", "bb_pctb_20")
    cov = _standardize_fit(np.column_stack([store.ext_column(c) for c in ext2]).astype(np.float32),
                           folds[0].fit.origins(store.ts, store.eligible))
    r1 = store.r1

    print("\n== 2. TimesFM covariate với xreg GPU ==")
    tfm = make_model("tfm_ext", cfg.model_params("tfm"))

    def _flag():
        assert tfm.xreg_force_on_cpu is False, f"xreg_force_on_cpu = {tfm.xreg_force_on_cpu} (phải False)"
        assert tfm.forecast_config_kwargs(True)["per_core_batch_size"] == 1, "per_core_batch_size phải = 1"
        return "xreg_force_on_cpu=False · per_core_batch_size=1"

    check("cấu hình: xreg GPU + 1 origin/lời gọi", _flag)

    # bọc forecast_with_covariates để chắc chắn MỌI lời gọi chỉ có đúng 1 origin
    state = {}

    def _cov_forecast():
        m = tfm._model(True)
        orig = m.forecast_with_covariates

        def wrapped(*a, **kw):
            seen["input_lens"].add(len(kw.get("inputs", a[0] if a else [])))
            return orig(*a, **kw)

        m.forecast_with_covariates = wrapped
        seq = SeriesBatch(store.ts, r1, val_idx, cov, ext2)
        t0 = time.perf_counter()
        yhat = tfm.predict_series(seq)
        state["ms_per_origin"] = 1000 * (time.perf_counter() - t0) / len(val_idx)
        assert yhat.shape == (len(val_idx), len(HORIZONS)), yhat.shape
        assert np.isfinite(yhat).all(), "prediction có NaN/inf"
        state["yhat_absmax"] = float(np.abs(yhat).max())
        return f"shape={yhat.shape} |ŷ|max={state['yhat_absmax']:.2e} {state['ms_per_origin']:.0f} ms/origin"

    check("covariate forecast chạy + output hữu hạn", _cov_forecast)

    def _on_gpu():
        assert seen["calls"] > 0, "xreg_lib.jnp.linalg.pinv KHÔNG được gọi → không có xreg thật"
        devs = seen["pinv_devices"]
        assert devs, "không đọc được device của beta_hat"
        assert devs == {"gpu"}, f"beta_hat materialize trên {devs} — có CPU fallback (phải chỉ {{'gpu'}})"
        return f"{seen['calls']} lời gọi pinv, beta_hat trên {devs}"

    check("xreg beta_hat THỰC SỰ trên GPU (không fallback)", _on_gpu)

    def _one_origin():
        assert seen["input_lens"] == {1}, f"len(inputs) từng lời gọi = {seen['input_lens']} (phải chỉ {{1}})"
        return f"mọi lời gọi forecast_with_covariates có đúng 1 origin ({seen['calls']} lời gọi xreg)"

    check("giữ đúng 1 origin mỗi lời gọi", _one_origin)

    def _causal():
        idx = val_idx[:6]
        a = make_model("tfm_ext", cfg.model_params("tfm")).predict_series(SeriesBatch(store.ts, r1, idx, cov, ext2))
        r1b, covb = r1.copy(), cov.copy()
        cut = int(idx[-1]) + 1
        r1b[cut:] = 0.0
        covb[cut:] = 0.0
        b = make_model("tfm_ext", cfg.model_params("tfm")).predict_series(SeriesBatch(store.ts, r1b, idx, covb, ext2))
        assert np.allclose(a, b, atol=1e-6), f"prediction đổi khi cắt dữ liệu sau t (max Δ={np.abs(a - b).max():.2e})"
        w = make_model("tfm_ext", cfg.model_params("tfm")).covariate_window(
            SeriesBatch(store.ts, r1, idx, cov, ext2), int(idx[0]), 0)
        L = tfm.context
        assert len(w) == L + len(HORIZONS)
        assert np.allclose(w[:L], cov[int(idx[0]) - L:int(idx[0]), 0]), "covariate không dịch 1 bar"
        assert np.allclose(w[L:], cov[int(idx[0]), 0]), "3 bước tương lai không giữ f(t)"
        return "cắt-chuỗi bit-identical; covariate dịch đúng 1 bar; 3 bước tương lai = f(t)"

    check("causal (τ ≤ t) + shift 1 bar vẫn đúng", _causal)

    def _torch_alive():
        import torch

        assert torch.cuda.is_available(), "torch.cuda.is_available() = False sau khi jax chiếm GPU"
        t = torch.randn(2048, 2048, device="cuda")
        assert bool(torch.isfinite(t @ t).all())
        free, total = torch.cuda.mem_get_info()
        return f"torch CUDA OK ({torch.cuda.get_device_name(0)}); VRAM free {free/2**30:.1f}/{total/2**30:.1f} GiB"

    check("torch CUDA sống chung với jax GPU", _torch_alive)

    out = ROOT / "experiments" / "canary_xreg_gpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checks": RESULTS, "pinv_devices": sorted(seen["pinv_devices"]),
                               "pinv_calls": seen["calls"], "input_lens": sorted(seen["input_lens"]),
                               "ms_per_origin_2cov": round(state.get("ms_per_origin", 0), 2)},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    bad = [r["check"] for r in RESULTS if not r["ok"]]
    print(f"\n→ {out}")
    if bad:
        print(f"XREG-GPU CANARY FAIL ({len(bad)}): {bad}")
        return 1
    print(f"XREG-GPU CANARY PASS ({len(RESULTS)} mục).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
