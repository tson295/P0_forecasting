"""§7.4 Inference latency — chỉ theo dõi. Pass riêng sau train: predict MỘT origin (batch 1), warm-up 50, p95/p99/max;
assert prediction batch == batch-1 (|Δ| ≤ 1e-6); không ảnh hưởng training/loss/quyết định.
Tree: 3 predictor độc lập → đo riêng từng h. Model một lần gọi ra 3 bước (LSTM head 3 output): predictor trả (n, 3) → shared = True.
Số thread: mặc định thư viện (batch 1, ghi cột `threads`); ghi train/predict device + phiên bản thư viện."""
from __future__ import annotations

import importlib
import time

import numpy as np
import pandas as pd

from .config import HORIZONS
from .harness import RunResult


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover
        pass


def lib_version(model) -> str:
    lib = getattr(model, "lib", "") or ""
    if not lib:
        return ""
    try:
        return f"{lib} {importlib.import_module(lib).__version__}"
    except Exception:  # pragma: no cover
        return lib


def measure_tabular(run: RunResult, warmup: int = 50, max_origins: int | None = None, atol: float = 1e-6, model=None) -> pd.DataFrame:
    """Đo trên VAL của fold đầu tiên có state. X_val là ma trận (tree) hoặc SeqBatch (LSTM: feats + idx)."""
    st = run.states[0]
    X = st.X_val
    is_seq = hasattr(X, "feats") and hasattr(X, "idx")
    n_all = len(X.idx) if is_seq else len(X)
    n = n_all if max_origins is None else min(n_all, max_origins)

    def sub(a: int, b: int):
        return type(X)(X.feats, X.idx[a:b]) if is_seq else X[a:b]

    meta = {"train_device": getattr(model, "train_device", ""), "predict_device": getattr(model, "predict_device", ""),
            "lib_version": lib_version(model) if model is not None else "", "threads": "lib_default"}
    rows = []
    for k, pred in enumerate(st.result.predictors):
        batch = np.asarray(pred(sub(0, n)), dtype=np.float64).reshape(n, -1)  # (n, 1) tree | (n, 3) shared
        for i in range(min(warmup, n)):
            pred(sub(i, i + 1))
        durs = np.empty(n)
        single = np.empty_like(batch)
        for i in range(n):
            x = sub(i, i + 1)
            _sync()
            t0 = time.perf_counter_ns()
            out = pred(x)
            _sync()
            durs[i] = (time.perf_counter_ns() - t0) / 1e6
            single[i] = np.asarray(out, dtype=np.float64).reshape(1, -1)[0]
        dev = float(np.max(np.abs(single - batch))) if n else 0.0
        assert dev <= atol, f"latency pass làm đổi prediction (batch != batch-1): max |Δ| = {dev:.3g} > {atol}"
        shared = bool(batch.shape[1] > 1)
        stat = {"n": n, "p95_ms": float(np.percentile(durs, 95)), "p99_ms": float(np.percentile(durs, 99)), "max_ms": float(durs.max()),
                "mean_ms": float(durs.mean()), "shared": shared, **meta}
        for h in (HORIZONS if shared else (k + 1,)):
            rows.append({"model": run.model, "h": int(h), **stat})
    return pd.DataFrame(rows)
