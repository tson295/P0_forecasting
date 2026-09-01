"""Fold-level parallelism CHỈ cho TimesFM (tfm_b0 / tfm_ext) — tối ưu THỰC THI, không đổi khoa học.

Vì sao chỉ TimesFM: profile 150 origin thật (B0*, 72 covariate) cho thấy 75,6% thời gian nằm ở forward
của TimesFM ở batch 1, GPU chỉ dùng 13–14% và 1,4/24 GiB → còn rất nhiều dư địa; các model tree/LSTM
đã nhanh sẵn nên không đụng tới.

Vì sao AN TOÀN về ngữ nghĩa: `harness.run_config` xử lý từng fold độc lập hoàn toàn — `_standardize_fit`
chỉ dùng `idx_fit` của chính fold đó, `TargetTransform.fit` cũng vậy, `_resolve_rounds` tra theo tên fold.
Phần duy nhất nằm ngoài vòng lặp (`feats_all`) là hàm tất định của (store, colset), không phụ thuộc fold.
Nên worker gọi ĐÚNG `run_config(store, model, colset, [folds[i]], ...)` — CÙNG một hàm, chỉ khác là
5 fold chạy ở 5 process — rồi parent ghép lại THEO ĐÚNG THỨ TỰ FOLD BAN ĐẦU.

KHÔNG đổi: checkpoint/config/head, backend + toán của xreg, 1 origin mỗi forecast_with_covariates,
dịch covariate 1 bar, feature set, thứ tự add-one (vẫn tuần tự — S phụ thuộc KEEP/DROP trước đó),
seed, ε, KEEP/DROP, prune PI, confirmation, định nghĩa/thứ tự fold, tfm-final/champion/ensemble/final.

Bật bằng biến môi trường `P0_TFM_FOLD_WORKERS` (mặc định 1 = TẮT, chạy y như cũ).
Chỉ áp dụng khi keep_states=False (calibrate, seed_noise, 39 candidate của add-one). Các run cần
`states` (prune PI, confirmation) vẫn chạy tuần tự trong parent vì FitResult giữ handle model sống.
"""
from __future__ import annotations

import atexit
import os
from multiprocessing import get_context

import numpy as np

_CTX: dict = {"cfg": None, "model": None, "workers": 1, "pool": None}
_W: dict = {}  # globals bên trong worker


def workers_configured() -> int:
    try:
        return max(1, int(os.environ.get("P0_TFM_FOLD_WORKERS", "1")))
    except ValueError:
        return 1


def configure(cfg, model) -> int:
    """Bật fold-parallel cho ĐÚNG object model này (chỉ TimesFM). Trả số worker thực tế."""
    n = workers_configured()
    if n <= 1 or getattr(model, "lib", "") != "timesfm":
        return 1
    _CTX.update(cfg=cfg, model=model, workers=n)
    return n


def active(model) -> bool:
    return _CTX["model"] is not None and model is _CTX["model"] and _CTX["workers"] > 1


# ------------------------------------------------------------------ worker
def _init(cfg, model):
    import warnings

    warnings.filterwarnings("ignore")
    from .cli import load_store

    store, folds, _final, _rep = load_store(cfg)
    _W["store"], _W["folds"], _W["model"] = store, folds, model


def _task(fold_i: int, colset_dict: dict, rounds, seed: int, want_yhat: bool):
    from .harness import ColSet, run_config

    cs = ColSet(tuple(colset_dict["b0"]), tuple(colset_dict["ext"]))
    # len(folds) == 1 → run_config đi nhánh tuần tự bình thường (không đệ quy vào pool)
    r = run_config(_W["store"], _W["model"], cs, [_W["folds"][fold_i]], rounds=rounds, seed=seed,
                   keep_states=want_yhat)
    yh = (np.asarray(r.states[0].yhat), np.asarray(r.states[0].idx_val)) if want_yhat else None
    return (fold_i, r.rmse[0], r.mae[0], r.r[0], r.dir_acc[0], r.e0[0], r.best_iters[0], r.rounds[0], yh)


def _pool():
    if _CTX["pool"] is None:
        ctx = get_context("spawn")  # spawn: parent đã init CUDA nên KHÔNG được fork
        _CTX["pool"] = ctx.Pool(_CTX["workers"], initializer=_init, initargs=(_CTX["cfg"], _CTX["model"]))
    return _CTX["pool"]


def shutdown():
    if _CTX["pool"] is not None:
        _CTX["pool"].terminate()
        _CTX["pool"].join()
        _CTX["pool"] = None


atexit.register(shutdown)


# ------------------------------------------------------------------ parent
def run_folds(model, colset, folds, rounds, seed, want_yhat: bool = False):
    """Chạy từng fold ở một process riêng rồi GHÉP THEO ĐÚNG THỨ TỰ FOLD. Trả RunResult như run_config."""
    from .harness import RunResult

    F = len(folds)
    args = [(i, colset.to_dict(), rounds, seed, want_yhat) for i in range(F)]
    out = _pool().starmap(_task, args)  # starmap giữ thứ tự, vẫn sắp lại theo fold_i cho chắc
    out = sorted(out, key=lambda t: t[0])
    if [t[0] for t in out] != list(range(F)):
        raise RuntimeError(f"fold-parallel trả thiếu/sai fold: {[t[0] for t in out]}")
    rmse, mae, rr, dacc, e0 = (np.zeros((F, 3)) for _ in range(5))
    best = np.zeros((F, 3), dtype=int)
    used, yhats = [], []
    for i, rm, ma, r_, da, ez, bi, rd, yh in out:
        rmse[i], mae[i], rr[i], dacc[i], e0[i], best[i] = rm, ma, r_, da, ez, bi
        used.append(tuple(int(x) for x in rd))
        yhats.append(yh)
    res = RunResult(getattr(model, "name", "?"), colset, seed, used, rmse, mae, rr, dacc, e0, best,
                    [f.name for f in folds], [])
    return (res, yhats) if want_yhat else res
