"""Fold-level parallelism cho MỌI model (§9, quyết định user 2026-09-03; mở rộng 2 GPU 2026-09-04c) — tối ưu THỰC THI, không đổi khoa học.

5 fold walk-forward là 5 cấu hình độc lập của cùng một (model, feature set): `harness.run_config` xử lý từng fold
hoàn toàn tách biệt (`_standardize_fit`/`TargetTransform.fit` chỉ dùng `idx_fit` của fold đó; `_resolve_rounds` tra
theo tên fold; `feats_all` là hàm tất định của (store, colset)). Worker gọi ĐÚNG `run_config(store, model, colset,
[fold], ...)` — cùng một hàm — rồi parent ghép lại THEO ĐÚNG THỨ TỰ FOLD BAN ĐẦU.

Từ 2026-09-04c module này là ADAPTER MỎNG trên `p0.scheduler.GpuScheduler`: mỗi worker process bị khoá vào ĐÚNG một
GPU vật lý (`CUDA_VISIBLE_DEVICES`), 5 fold của một cấu hình được rải ĐỘNG lên GPU nào rảnh trước (không affinity theo
model family), nhiều branch model độc lập có thể cùng đưa task vào hàng đợi. Số worker = len(gpu_devices) ×
gpu_slots_per_device (mặc định 2 GPU × 1 slot = 2 task nặng đồng thời); `P0_FOLD_WORKERS` vẫn ghi đè tổng số worker.

Bất biến giữ nguyên: seed, số vòng, ε, KEEP/DROP, prune PI, confirmation, định nghĩa fold, champion/ensemble/final;
thứ tự candidate vẫn TUẦN TỰ (S đổi sau mỗi KEEP) — chỉ các fold của một candidate chạy song song.
Không có CPU fallback: worker dựng model bằng chính `cli.model_for(cfg, name, allow_cpu)` (GPU trên data thật);
hết VRAM → task fail rõ ràng, user giảm `gpu_slots_per_device` / `P0_FOLD_WORKERS`.

Bật: `gpu_devices`/`gpu_slots_per_device` trong config, env `P0_GPU_DEVICES`/`P0_GPU_SLOTS_PER_DEVICE`, hoặc
`P0_FOLD_WORKERS`. Một worker (1 GPU, 1 slot) = chạy tuần tự trong chính process cha, y hệt hành vi cũ.
"""
from __future__ import annotations

import atexit
import threading

import numpy as np

from . import gpu, scheduler

_LOCK = threading.RLock()
_CTX: dict = {"cfg": None, "allow_cpu": False, "workers": 1, "sched": None}
POOL_MARK = "_p0_pool"  # `cli.model_for` gắn tên model → chỉ model dựng lại được từ (cfg, name) mới đi qua scheduler


def workers_configured(cfg=None) -> int:
    """Số worker GPU = len(gpu_devices) × gpu_slots_per_device (env ưu tiên); 1 = tắt (chạy trong process cha)."""
    return gpu.worker_slots(cfg)[2]


def configure(cfg, model=None, name: str | None = None, allow_cpu: bool = False) -> int:
    """Bật scheduler GPU cho config này (idempotent, dùng chung cho MỌI branch model). Trả số worker thực tế (1 = tắt)."""
    n = workers_configured(cfg)
    with _LOCK:
        if n <= 1 or scheduler.in_worker():
            shutdown()
            _CTX.update(cfg=cfg, allow_cpu=allow_cpu, workers=1)
            return 1
        if _CTX["sched"] is not None and (_CTX["cfg"] is not cfg or _CTX["allow_cpu"] != allow_cpu):
            shutdown()
        _CTX.update(cfg=cfg, allow_cpu=allow_cpu, workers=n)
        if _CTX["sched"] is None:
            _CTX["sched"] = scheduler.GpuScheduler(cfg, allow_cpu=allow_cpu, exp_dir=getattr(cfg, "exp_dir", None)).start()
    return n


def sched() -> scheduler.GpuScheduler | None:
    return _CTX["sched"]


def active(model=None) -> bool:
    """True khi task nên đi qua scheduler: pool đang chạy, không ở trong worker, và model dựng lại được từ (cfg, name)."""
    if _CTX["sched"] is None or _CTX["workers"] <= 1 or scheduler.in_worker():
        return False
    return model is None or getattr(model, POOL_MARK, None) is not None


def model_name(model) -> str:
    n = getattr(model, POOL_MARK, None)
    if n is None:
        raise RuntimeError(f"model {getattr(model, 'name', '?')} không dựng lại được từ config → không đưa vào scheduler")
    return str(n)


def shutdown() -> None:
    with _LOCK:
        if _CTX["sched"] is not None:
            _CTX["sched"].shutdown()
            _CTX["sched"] = None
        _CTX["workers"] = 1


atexit.register(shutdown)


# ------------------------------------------------------------------ parent API
def run_folds(store, model, colset, folds, rounds, seed, want_yhat: bool = False, latency_origins=None):
    """Chạy từng fold ở một worker GPU rồi GHÉP THEO ĐÚNG THỨ TỰ FOLD. Trả RunResult như run_config.

    want_yhat=True → states "nhẹ" (idx_val + ŷ, không có predictor sống) đủ cho confirmation/ensemble/artifact;
    latency_origins ≠ None → worker của fold ĐẦU đo latency §7.4 (predictor sống ở trong worker) → `RunResult.latency`.
    """
    from .harness import FoldState, RunResult

    name = model_name(model)
    F = len(folds)
    tasks = [scheduler.Task(kind="run_fold", model=name, fold=f.name, seed=int(seed),
                            payload={"model": name, "colset": colset.to_dict(), "fold": f.name, "rounds": rounds, "seed": int(seed),
                                     "want_yhat": bool(want_yhat),
                                     "latency_origins": (latency_origins if (i == 0 and latency_origins is not None) else None)})
             for i, f in enumerate(folds)]
    out = _CTX["sched"].submit(tasks)
    by_name = {r["fold"]: r for r in out}
    if sorted(by_name) != sorted(f.name for f in folds) or len(out) != F:
        raise RuntimeError(f"scheduler trả thiếu/sai fold: {[r['fold'] for r in out]}")
    rmse, mae, rr, dacc, e0 = (np.zeros((F, 3)) for _ in range(5))
    best = np.zeros((F, 3), dtype=int)
    used, states, latency, aux = [], [], None, []
    for i, f in enumerate(folds):
        r = by_name[f.name]
        rmse[i], mae[i], rr[i], dacc[i], e0[i], best[i] = r["rmse"], r["mae"], r["r"], r["dir_acc"], r["e0"], r["best_iters"]
        used.append(tuple(int(x) for x in r["rounds"]))
        aux.append(r.get("aux"))
        if want_yhat:
            idx_val, yhat = r["yhat"]
            states.append(FoldState(f, f.fit.origins(store.ts, store.eligible), f.es.origins(store.ts, store.eligible),
                                    idx_val, None, None, None, yhat))
        if r.get("latency") is not None:
            latency = r["latency"]
    return RunResult(getattr(model, "name", "?"), colset, seed, used, rmse, mae, rr, dacc, e0, best,
                     [f.name for f in folds], states, latency, aux if any(a is not None for a in aux) else None)


def run_prune_pi(store, model, colset, folds, rounds, seed: int, repeats: int = 3):
    """Prune PI (§2.1a) chạy TRỌN VẸN trong MỘT worker GPU: cùng một dòng RNG cho mọi fold như bản tuần tự
    (`filter_b0.permutation_importance` tạo `default_rng(seed)` một lần) → PI không đổi một chữ số."""
    import pandas as pd

    from .harness import ColSet

    name = model_name(model)
    t = scheduler.Task(kind="prune_pi", model=name, seed=int(seed),
                       payload={"model": name, "colset": colset.to_dict(), "folds": [f.name for f in folds],
                                "rounds": rounds, "seed": int(seed), "repeats": int(repeats)})
    res = _CTX["sched"].submit([t])[0]
    df = pd.DataFrame(res["pi"], columns=res["columns"])
    pruned = ColSet(colset.b0, tuple(res["pruned_ext"]), colset.locked_ext)
    return pruned, df


def submit(tasks: list[scheduler.Task]):
    """Đưa một nhóm task tuỳ ý (autots bake-off / scoring / probe) lên scheduler; kết quả theo đúng thứ tự task."""
    return _CTX["sched"].submit(tasks)
