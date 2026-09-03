"""Fold-level parallelism cho MỌI model (§9, quyết định user 2026-09-03) — tối ưu THỰC THI, không đổi khoa học.

5 fold walk-forward là 5 cấu hình độc lập của cùng một (model, feature set): `harness.run_config` xử lý từng fold
hoàn toàn tách biệt (`_standardize_fit`/`TargetTransform.fit` chỉ dùng `idx_fit` của fold đó; `_resolve_rounds` tra
theo tên fold; `feats_all` là hàm tất định của (store, colset)). Worker gọi ĐÚNG `run_config(store, model, colset,
[fold], ...)` — cùng một hàm — rồi parent ghép lại THEO ĐÚNG THỨ TỰ FOLD BAN ĐẦU.

Bất biến giữ nguyên: seed, số vòng, ε, KEEP/DROP, prune PI, confirmation, định nghĩa fold, champion/ensemble/final;
thứ tự candidate vẫn TUẦN TỰ (S đổi sau mỗi KEEP) — chỉ các fold của một candidate chạy song song.
Không có CPU fallback: worker dựng model bằng chính `cli.model_for(cfg, name, allow_cpu)` (GPU trên data thật);
hết VRAM → worker raise → lệnh dừng rõ ràng, user giảm `P0_FOLD_WORKERS` / `fold_workers`.

Bật: biến môi trường `P0_FOLD_WORKERS` (ưu tiên) hoặc `fold_workers` trong config (mặc định 1 = TẮT, chạy y như cũ).
Áp dụng khi keep_states=False (calibrate, ε, add-one) và khi caller chỉ cần prediction (`parallel_ok`, confirmation:
worker trả (idx_val, ŷ) + best_iters, đo latency §7.4 ngay trong worker ở fold đầu). Run cần predictor sống
(prune PI, filter-b0) chạy tuần tự trong parent.
"""
from __future__ import annotations

import atexit
import os
from multiprocessing import get_context

import numpy as np

_CTX: dict = {"cfg": None, "model": None, "name": None, "allow_cpu": False, "workers": 1, "pool": None}
_W: dict = {}  # globals bên trong worker


def workers_configured(cfg=None) -> int:
    env = os.environ.get("P0_FOLD_WORKERS")
    try:
        if env is not None:
            return max(1, int(env))
        return max(1, int(getattr(cfg, "fold_workers", 1) or 1))
    except ValueError:
        return 1


def configure(cfg, model, name: str, allow_cpu: bool = False) -> int:
    """Bật fold-parallel cho ĐÚNG object model này. Trả số worker thực tế (1 = tắt)."""
    n = workers_configured(cfg)
    if n <= 1:
        _CTX.update(model=None, workers=1)
        return 1
    if _CTX["pool"] is not None and (_CTX["name"] != name or _CTX["allow_cpu"] != allow_cpu or _CTX["cfg"] is not cfg):
        shutdown()
    _CTX.update(cfg=cfg, model=model, name=name, allow_cpu=allow_cpu, workers=n)
    return n


def active(model) -> bool:
    return _CTX["model"] is not None and model is _CTX["model"] and _CTX["workers"] > 1


# ------------------------------------------------------------------ worker
def _init(cfg, name: str, allow_cpu: bool):
    import warnings

    warnings.filterwarnings("ignore")
    from .cli import load_store, model_for

    store, folds, final, _rep = load_store(cfg)
    _W.update(store=store, folds={f.name: f for f in folds + [final]}, model=model_for(cfg, name, allow_cpu))


def _task(fold_name: str, colset_dict: dict, rounds, seed: int, want_yhat: bool, latency_origins):
    from .harness import ColSet, run_config

    cs = ColSet.from_dict(colset_dict)
    fold = _W["folds"][fold_name]
    # len(folds) == 1 → run_config đi nhánh tuần tự bình thường (không đệ quy vào pool)
    r = run_config(_W["store"], _W["model"], cs, [fold], rounds=rounds, seed=seed, keep_states=bool(want_yhat or latency_origins is not None))
    yh = (np.asarray(r.states[0].idx_val), np.asarray(r.states[0].yhat)) if want_yhat else None
    lat = None
    if latency_origins is not None:
        from .latency import measure_tabular

        lat = measure_tabular(r, warmup=50, max_origins=latency_origins, model=_W["model"]).to_dict("records")
    return (fold_name, r.rmse[0], r.mae[0], r.r[0], r.dir_acc[0], r.e0[0], r.best_iters[0], r.rounds[0], yh, lat)


def _pool():
    if _CTX["pool"] is None:
        ctx = get_context("spawn")  # spawn: parent có thể đã init CUDA nên KHÔNG được fork
        _CTX["pool"] = ctx.Pool(_CTX["workers"], initializer=_init, initargs=(_CTX["cfg"], _CTX["name"], _CTX["allow_cpu"]))
    return _CTX["pool"]


def shutdown():
    if _CTX["pool"] is not None:
        _CTX["pool"].terminate()
        _CTX["pool"].join()
        _CTX["pool"] = None


atexit.register(shutdown)


# ------------------------------------------------------------------ parent
def run_folds(store, model, colset, folds, rounds, seed, want_yhat: bool = False, latency_origins=None):
    """Chạy từng fold ở một process riêng rồi GHÉP THEO ĐÚNG THỨ TỰ FOLD. Trả RunResult như run_config.

    want_yhat=True → states "nhẹ" (idx_val + ŷ, không có predictor sống) đủ cho confirmation/ensemble/artifact;
    latency_origins ≠ None → worker của fold ĐẦU đo latency §7.4 (predictor sống ở trong worker) → `RunResult.latency`.
    """
    from .harness import FoldState, RunResult

    F = len(folds)
    args = [(f.name, colset.to_dict(), rounds, seed, want_yhat, latency_origins if (i == 0 and latency_origins is not None) else None)
            for i, f in enumerate(folds)]
    out = _pool().starmap(_task, args)  # starmap giữ thứ tự; vẫn sắp lại theo tên fold cho chắc
    by_name = {t[0]: t for t in out}
    if sorted(by_name) != sorted(f.name for f in folds) or len(out) != F:
        raise RuntimeError(f"fold-parallel trả thiếu/sai fold: {[t[0] for t in out]}")
    rmse, mae, rr, dacc, e0 = (np.zeros((F, 3)) for _ in range(5))
    best = np.zeros((F, 3), dtype=int)
    used, states, latency = [], [], None
    for i, f in enumerate(folds):
        _, rm, ma, r_, da, ez, bi, rd, yh, lat = by_name[f.name]
        rmse[i], mae[i], rr[i], dacc[i], e0[i], best[i] = rm, ma, r_, da, ez, bi
        used.append(tuple(int(x) for x in rd))
        if want_yhat:
            idx_val, yhat = yh
            states.append(FoldState(f, f.fit.origins(store.ts, store.eligible), f.es.origins(store.ts, store.eligible),
                                    idx_val, None, None, None, yhat))
        if lat is not None:
            latency = lat
    return RunResult(getattr(model, "name", "?"), colset, seed, used, rmse, mae, rr, dacc, e0, best,
                     [f.name for f in folds], states, latency)
