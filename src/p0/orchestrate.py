"""Orchestrator DAG cho vòng expanded-data (§9/§14 hiệu chỉnh 2026-09-04c) — CHỈ điều phối thực thi.

Vấn đề: các nhánh model độc lập nhau về khoa học, nhưng nếu chạy tuần tự thì khi một nhánh chỉ còn 1 task, GPU còn
lại nằm không. Orchestrator cho nhiều nhánh cùng đưa task vào scheduler; hai GPU luôn có việc khi có ≥ 2 task sẵn sàng.

    loop lgbm ─┐
    loop xgb  ─┤
    loop cat  ─┤
    loop xgbrf─┤                       (mỗi nhánh: candidate TUẦN TỰ, chỉ fold song song)
    loop lstm ─┤
    loop tfm  ─┴─→ tfm-final ─┐
    loop autots_wr ─┐         │
    loop autots_mr ─┴─→ autots-search ─┤
                                       ↓
                       CHAMPION REPLAY (thứ tự CỐ ĐỊNH lgbm→xgb→cat→tfm→xgbrf→autots→lstm, chỉ đọc artifact)
                                       ↓
                                    ensemble
                                       ↓
                          `final` (TEST) VẪN LÀ LỆNH RIÊNG — orchestrator không bao giờ chạm TEST

Bất biến: thứ tự hoàn thành của nhánh KHÔNG ảnh hưởng champion (replay theo thứ tự methodology cố định); mỗi nhánh
có state/artifact riêng; seed, ε, KEEP/DROP, prune, confirmation, champion rule không đổi một dòng.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import fold_parallel, gpu, scheduler
from .checker_log import record as ck_record
from .config import RunConfig

LOOP_MODELS = ("lgbm", "xgb", "cat", "tfm", "xgbrf", "autots_wr", "autots_mr", "lstm")


@dataclass
class Branch:
    name: str
    deps: tuple[str, ...] = ()
    kind: str = "loop"          # loop | tfm-final | autots-search
    model: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    status: str = "pending"
    error: str | None = None
    thread: threading.Thread | None = field(default=None, repr=False)


def build_dag(models: list[str]) -> list[Branch]:
    """DAG nhánh theo phụ thuộc KHOA HỌC (không phải theo GPU): probe → bước gộp đại diện."""
    branches: list[Branch] = []
    for m in models:
        branches.append(Branch(name=f"loop:{m}", kind="loop", model=m))
    if "tfm" in models:
        branches.append(Branch(name="tfm-final", deps=("loop:tfm",), kind="tfm-final", model="tfm"))
    if "autots_wr" in models and "autots_mr" in models:
        branches.append(Branch(name="autots-search", deps=("loop:autots_wr", "loop:autots_mr"), kind="autots-search", model="autots"))
    return branches


def _ns(args, **over) -> argparse.Namespace:
    base = {"smoke": bool(getattr(args, "smoke", False)), "allow_cpu": bool(getattr(args, "allow_cpu", False)),
            "max_candidates": getattr(args, "max_candidates", None), "no_standalone": bool(getattr(args, "no_standalone", False)),
            "latency_origins": getattr(args, "latency_origins", None), "resume": bool(getattr(args, "resume", False)),
            "config": getattr(args, "config", None)}
    return argparse.Namespace(**{**base, **over})


def run_dag(branches: list[Branch], run_branch, max_active: int, on_event=None) -> list[Branch]:
    """Chạy DAG: nhánh nào đủ dep thì khởi động (tối đa `max_active` nhánh song song), thread chỉ điều phối —
    mọi việc nặng nằm ở worker GPU. Trả về danh sách nhánh đã cập nhật trạng thái.

    `run_branch(branch)` thực thi nhánh (đồng bộ). Lỗi ở một nhánh: dừng khởi động nhánh mới, chờ nhánh đang chạy,
    rồi raise — không nhánh nào bị bỏ lửng giữa chừng.
    """
    by_name = {b.name: b for b in branches}
    lock = threading.Lock()
    cv = threading.Condition(lock)
    active: dict[str, Branch] = {}
    failed: list[Branch] = []

    def worker(b: Branch) -> None:
        try:
            run_branch(b)
            with cv:
                b.status, b.ended_at = "done", time.time()
        except BaseException as e:  # noqa: BLE001 — lỗi nhánh phải nổi lên rõ ràng
            with cv:
                b.status, b.ended_at, b.error = "error", time.time(), f"{type(e).__name__}: {e}"
                failed.append(b)
        finally:
            with cv:
                active.pop(b.name, None)
                cv.notify_all()
            if on_event:
                on_event(b)

    def ready(b: Branch) -> bool:
        return b.status == "pending" and all(by_name[d].status == "done" for d in b.deps)

    with cv:
        while True:
            if failed:
                while active:
                    cv.wait(0.2)
                break
            for b in branches:  # thứ tự khởi động = thứ tự DAG (tất định); ai xong trước không đổi kết quả
                if len(active) >= max_active:
                    break
                if ready(b):
                    b.status, b.started_at = "running", time.time()
                    b.thread = threading.Thread(target=worker, args=(b,), name=f"p0-branch-{b.name}", daemon=True)
                    active[b.name] = b
                    b.thread.start()
                    if on_event:
                        on_event(b)
            if not active and not any(ready(b) for b in branches):
                break
            cv.wait(0.2)
    for b in branches:
        if b.thread is not None:
            b.thread.join()
    if failed:
        raise RuntimeError("; ".join(f"{b.name}: {b.error}" for b in failed))
    return branches


def cmd_orchestrate(cfg: RunConfig, args) -> None:
    """Chạy toàn bộ vòng tìm kiếm trên 2 GPU đối xứng rồi champion replay + ensemble. KHÔNG chạm TEST (`final` riêng)."""
    from .cli import (CHAMPION_ORDER, champion_deferred, cmd_autots_search, cmd_champion_replay, cmd_ensemble, cmd_loop,
                      cmd_tfm_final, say)

    models = [m.strip() for m in str(args.models).split(",") if m.strip()] if getattr(args, "models", None) else \
        [m for m in (cfg.model_order or list(LOOP_MODELS))]
    bad = [m for m in models if m not in LOOP_MODELS]
    if bad:
        raise SystemExit(f"orchestrate: model không hợp lệ {bad} (hợp lệ: {list(LOOP_MODELS)})")
    branches = build_dag(models)
    devices, slots, n_workers = gpu.worker_slots(cfg)
    max_active = int(getattr(args, "max_branches", None) or getattr(cfg, "max_branches", 0) or n_workers)
    say(f"orchestrate: {len(branches)} nhánh trên {n_workers} worker GPU (devices {devices} × {slots} slot), "
        f"tối đa {max_active} nhánh đồng thời; champion HOÃN đến replay (thứ tự cố định {list(CHAMPION_ORDER)})")
    for b in branches:
        say(f"  {b.name:<20} deps={list(b.deps) or '—'}")
    if getattr(args, "dry_run", False):
        return
    if not champion_deferred(cfg):
        raise SystemExit("orchestrate: cần `defer_champion: true` trong config (hoặc P0_DEFER_CHAMPION=1) — nhánh chạy song song thì "
                         "champion phải được replay theo thứ tự cố định (§14).")
    exp = cfg.exp_dir
    exp.mkdir(parents=True, exist_ok=True)
    fold_parallel.configure(cfg, None, "orchestrate", bool(getattr(args, "allow_cpu", False)))
    log_path = exp / "orchestrate_log.jsonl"

    def event(b: Branch) -> None:
        row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "branch": b.name, "status": b.status, "model": b.model,
               "deps": list(b.deps), "started_at": b.started_at, "ended_at": b.ended_at,
               "duration_sec": (round(b.ended_at - b.started_at, 2) if (b.ended_at and b.started_at) else None), "error": b.error}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        say(f"[branch] {b.name} → {b.status}" + (f" ({row['duration_sec']}s)" if row["duration_sec"] else ""))

    def run_branch(b: Branch) -> None:
        scheduler.set_branch(b.name)  # nhãn branch cho chính sách round-robin của scheduler + scheduler_log
        if b.kind == "loop":
            cmd_loop(cfg, _ns(args, model=b.model))
        elif b.kind == "tfm-final":
            cmd_tfm_final(cfg, _ns(args))
        elif b.kind == "autots-search":
            cmd_autots_search(cfg, _ns(args))
        else:
            raise KeyError(b.kind)

    t0 = time.time()
    try:
        run_dag(branches, run_branch, max_active=max_active, on_event=event)
    except BaseException as e:
        ck_record(exp, "orchestrate", "ERROR", "BRANCH_FAILED", f"nhánh lỗi: {e}")
        raise
    say(f"mọi nhánh xong sau {time.time() - t0:.0f}s → champion replay (chỉ đọc artifact)")
    scheduler.set_branch("champion-replay")
    cmd_champion_replay(cfg, _ns(args, allow_partial=bool(getattr(args, "allow_partial", False)),
                                 force_replay=bool(getattr(args, "force_replay", False))))
    if not getattr(args, "skip_ensemble", False):
        scheduler.set_branch("ensemble")
        cmd_ensemble(cfg, _ns(args))
    fold_parallel.shutdown()
    util = summarize_utilization(exp)  # §20: GPU nào bận bao lâu → chỉnh scheduling sau, KHÔNG dùng để đổi methodology
    busy = ", ".join(f"GPU {g}: {sec / 60:.1f} phút" for g, sec in sorted(util["busy_sec_by_gpu"].items()))
    say(f"thời gian bận theo GPU ({util['tasks']} task): {busy or 'không có task nào qua scheduler'}")
    ck_record(exp, "orchestrate", "PASS", "ORCHESTRATE", f"{len(branches)} nhánh + champion replay xong trong {time.time() - t0:.0f}s "
              f"({n_workers} worker GPU {devices}); bận theo GPU: {util['busy_sec_by_gpu']}; TEST không bị chạm (chạy `final` riêng)")
    say(f"orchestrate xong ({time.time() - t0:.0f}s). TEST chưa chạm — chạy `python run.py final` khi đã sẵn sàng.")


def summarize_utilization(exp_dir: Path) -> dict:
    """Đọc scheduler_log.jsonl → thời gian bận của từng GPU (dùng để chỉnh scheduling sau, không đổi methodology)."""
    rows = scheduler.read_scheduler_log(exp_dir)
    busy: dict[int, float] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        busy[int(r.get("gpu_physical_id", -1))] = busy.get(int(r.get("gpu_physical_id", -1)), 0.0) + float(r.get("duration_sec") or 0)
    return {"tasks": len(rows), "busy_sec_by_gpu": busy}
