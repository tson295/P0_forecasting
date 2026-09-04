"""Scheduler GPU động, ĐỐI XỨNG giữa các GPU (§9 hiệu chỉnh 2026-09-04c) — CHỈ tối ưu thực thi.

    READY TASK QUEUE (round-robin giữa branch, FIFO trong branch)
                      │
            ┌─────────┴─────────┐
            ↓                   ↓
        GPU worker 0        GPU worker 1
      CUDA_VISIBLE=0        CUDA_VISIBLE=1
            └──── xong ─────────┘
                      ↓
        kết quả ghép theo THỨ TỰ TASK BAN ĐẦU (tất định)

Bất biến khoa học KHÔNG đổi: seed, số vòng/epoch, ε, KEEP/DROP, prune PI, confirmation, champion, ensemble, split, TEST.
Task nào chạy trên GPU nào, và branch nào xong trước, KHÔNG ảnh hưởng kết quả: mỗi task là một hàm tất định của
(dataset, model, colset, fold, rounds, seed) và parent luôn ghép lại theo thứ tự cố định.

Quy tắc (yêu cầu user 2026-09-04c):
- Không GPU nào mang vai trò "ML"/"DL"; không pin model family vào GPU (§6). Worker là như nhau, ai rảnh nhận task kế tiếp.
- Mặc định MỘT task nặng / MỘT GPU vật lý (`gpu_slots_per_device = 1`) — không tự ý oversubscribe VRAM (§8).
- OOM / lỗi backend → task fail rõ ràng, KHÔNG CPU fallback, không đổi methodology.
- 5 fold của một candidate được rải động lên các GPU; candidate thì vẫn TUẦN TỰ (S đổi sau KEEP) — do caller giữ.
- Mọi task ghi một dòng `experiments/<run>/scheduler_log.jsonl` (§20).
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from . import gpu

SCHED_LOG = "scheduler_log.jsonl"
_LOCAL = threading.local()  # branch/stage/candidate của thread hiện tại (chỉ để log + chính sách công bằng)


# ----------------------------------------------------------------------------- context (chỉ ảnh hưởng log/fairness)
def set_branch(name: str) -> None:
    _LOCAL.branch = str(name)


def current_branch() -> str:
    return getattr(_LOCAL, "branch", "main")


class stage:
    """`with scheduler.stage("add_one", candidate=...)` — gắn nhãn cho mọi task submit trong khối (chỉ để log)."""

    def __init__(self, name: str, **fields):
        self.ctx = {"stage": name, **fields}

    def __enter__(self):
        self.prev = getattr(_LOCAL, "ctx", {})
        _LOCAL.ctx = {**self.prev, **self.ctx}
        return self

    def __exit__(self, *exc):
        _LOCAL.ctx = self.prev
        return False


def current_ctx() -> dict:
    return dict(getattr(_LOCAL, "ctx", {}))


# ----------------------------------------------------------------------------- task
@dataclass
class Task:
    kind: str
    payload: dict
    stage: str = ""
    model: str = ""
    fold: str = ""
    seed: int | None = None
    candidate: str = ""
    branch: str = ""
    task_id: str = ""


@dataclass
class _Group:
    tasks: list[Task]
    results: list[Any]
    errors: list[str | None]
    remaining: int
    done: threading.Event = field(default_factory=threading.Event)


# ----------------------------------------------------------------------------- worker
def _precompute_ext(cfg) -> dict:
    """Tính TRƯỚC, ĐÚNG MỘT LẦN, TRONG PROCESS CHA — trước khi bất kỳ GPU worker nào được spawn: hợp cột ext của
    S0_m + Candidate_m cho các model trong `cfg.model_order` (chỉ model đã lock-s0 — xem `s0.union_ext_columns`),
    gọi `Store.ensure_ext` MỘT LẦN rồi trả lại đúng các mảng đã tính. `GpuScheduler.start()` gọi hàm này một lần
    (không phải một lần mỗi worker) rồi truyền dict kết quả cho MỌI worker qua `_worker_main` — mỗi worker CHỈ nạp
    lại (`store._ext.update(...)`), không tự gọi `compute_short`/`compute_ext` nữa (§ run ML+LSTM).

    KHÔNG bỏ qua lỗi: `load_store`/`union_ext_columns`/`ensure_ext` lỗi ở đây (checksum/data hỏng, cột không có
    định nghĩa, ...) PHẢI nổi lên rõ ràng và dừng NGAY — TRƯỚC KHI bất kỳ worker nào được spawn. Không có nhánh
    "trả {} rồi để worker/candidate search tự tính lại on-demand" — nếu cần precompute mà tính không ra, đó là lỗi
    của run, không phải sự cố tài nguyên GPU để hoãn cho worker xử lý."""
    from .cli import load_store
    from .s0 import union_ext_columns

    store, _, _, _ = load_store(cfg)
    cols = union_ext_columns(cfg.exp_dir, getattr(cfg, "model_order", None) or (), str(cfg.dataset_label))
    if not cols:
        return {}
    store.ensure_ext(cols)
    missing = [c for c in cols if c not in store._ext]
    if missing:
        raise RuntimeError(f"precompute ext: thiếu cột sau khi ensure_ext (không thể xảy ra nếu ensure_ext đúng): {missing}")
    return {c: store._ext[c] for c in cols}


def _worker_store(cfg, precomputed_ext: dict | None):
    """Dựng Store của MỘT worker (B0/grid không tránh được — mỗi worker một process riêng) rồi NẠP THẲNG
    `precomputed_ext` (đã tính một lần trong process cha, xem `_precompute_ext`) vào cache ext — worker KHÔNG tự
    gọi `compute_short`/`compute_ext` cho các cột đã có sẵn ở đây."""
    from .cli import load_store

    store, folds, final, _ = load_store(cfg)
    if precomputed_ext:
        store._ext.update(precomputed_ext)
    return store, folds, final


def _worker_main(worker_id: int, device: int, task_q, result_q, cfg, allow_cpu: bool, require_gpu: bool, light: bool = False,
                 precomputed_ext: dict | None = None) -> None:
    """Process worker gắn CHẶT vào một GPU vật lý. Bind device TRƯỚC mọi import CUDA (§17, §18)."""
    info = gpu.bind_worker_device(worker_id, device)
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from .cli import model_for, set_say_prefix  # import sau khi đã bind device

        set_say_prefix(f"gpu{device}")
        rep = gpu.device_report(require_gpu=bool(require_gpu and not allow_cpu))
        W = {"cfg": cfg, "allow_cpu": allow_cpu, "models": {}, "model_for": model_for, "grid": {}}
        if not light:  # `gpu-probe` chỉ kiểm thiết bị → không đọc data (nhanh, không cần checksum)
            store, folds, final = _worker_store(cfg, precomputed_ext)
            # CHỈ nạp fold walk-forward (VAL). Fold `final` (TEST) KHÔNG bao giờ vào worker: TEST chỉ được chạm ở
            # `run.py final` (tuần tự, có TEST_SENTINEL) — scheduler không được là đường vòng qua bất biến TEST-một-lần.
            W.update(store=store, folds={f.name: f for f in folds}, fold_order=[f.name for f in folds],
                     test_fold=str(final.name))
        result_q.put({"type": "ready", "worker_id": worker_id, "device": device, "report": {**info, **rep}})
    except BaseException as e:  # init hỏng (thiếu GPU, checksum, thư viện) → báo rồi thoát, KHÔNG fallback
        result_q.put({"type": "init_error", "worker_id": worker_id, "device": device,
                      "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
        return
    while True:
        item = task_q.get()
        if item is None:
            return
        task_id, kind, payload = item["task_id"], item["kind"], item["payload"]
        t0 = time.time()
        try:
            value, err = execute(kind, payload, W), None
        except SystemExit as e:
            value, err = None, f"SystemExit: {e}"
        except BaseException as e:
            value, err = None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result_q.put({"type": "result", "worker_id": worker_id, "device": device, "task_id": task_id, "value": value,
                      "error": err, "started_at": t0, "ended_at": time.time(), "peak_vram_mb": gpu.peak_vram_mb()})


def _model(W: dict, name: str):
    """Model trong worker: dựng ĐÚNG như parent (`cli.model_for`) và cache lại (§4: không init lặp lại)."""
    if name not in W["models"]:
        W["models"][name] = W["model_for"](W["cfg"], name, W["allow_cpu"])
    return W["models"][name]


def _colset(d: dict):
    from .harness import ColSet

    return ColSet.from_dict(d)


def _fold(W: dict, name: str):
    """Fold của task — chặn cứng mọi tham chiếu tới fold TEST: TEST chỉ chạm ở `final` (sentinel một lần)."""
    if name not in W["folds"]:
        if name == W.get("test_fold"):
            raise RuntimeError(f"scheduler: task yêu cầu fold TEST ({name}) — TEST chỉ được chạm ở `run.py final` "
                               "(TEST_SENTINEL một lần), không bao giờ qua scheduler.")
        raise KeyError(f"scheduler: fold không hợp lệ {name!r} (hợp lệ: {W.get('fold_order')})")
    return W["folds"][name]


def execute(kind: str, p: dict, W: dict):
    """Thực thi một task trong worker. MỌI kind gọi lại đúng hàm khoa học đang dùng ở bản tuần tự."""
    import numpy as np

    if kind == "device_report":
        return gpu.device_report(require_gpu=False)
    if kind == "probe":  # chỉ dùng cho `gpu-probe` và test scheduler — không phải training
        time.sleep(float(p.get("sleep_ms", 0)) / 1000.0)
        return {**gpu.device_report(require_gpu=False), "tag": p.get("tag")}
    if kind == "backend_probe":  # §11: phép tính GPU nhỏ THẬT cho từng backend, NGAY TRONG worker đã mask
        rep = gpu.device_report(require_gpu=False)
        res = gpu.backend_probe(tuple(p.get("backends") or gpu.BACKENDS), str(p.get("lgbm_device_type", "cuda")))
        return {"report": rep, "backends": res}
    if kind == "run_fold":
        from .harness import run_config

        model, cs, fold = _model(W, p["model"]), _colset(p["colset"]), _fold(W, p["fold"])
        want_yhat, lat_origins = bool(p.get("want_yhat")), p.get("latency_origins")
        r = run_config(W["store"], model, cs, [fold], rounds=p.get("rounds"), seed=int(p["seed"]),
                       keep_states=bool(want_yhat or lat_origins is not None))
        yh = (np.asarray(r.states[0].idx_val), np.asarray(r.states[0].yhat)) if want_yhat else None
        lat = None
        if lat_origins is not None:
            from .latency import measure_tabular

            lat = measure_tabular(r, warmup=50, max_origins=lat_origins, model=model).to_dict("records")
        return {"fold": p["fold"], "rmse": r.rmse[0], "mae": r.mae[0], "r": r.r[0], "dir_acc": r.dir_acc[0], "e0": r.e0[0],
                "best_iters": r.best_iters[0], "rounds": r.rounds[0], "yhat": yh, "latency": lat,
                "aux": (r.aux[0] if r.aux else None)}
    if kind == "prune_pi":
        # CẢ job (5 fold + vòng PI) chạy trong MỘT process: giữ nguyên một dòng RNG duy nhất như bản tuần tự
        # (`filter_b0.permutation_importance` dùng chung `default_rng(seed)` qua các fold) → số PI không đổi.
        from .loop import prune_pi

        model, cs = _model(W, p["model"]), _colset(p["colset"])
        folds = [_fold(W, n) for n in p["folds"]]
        pruned, df = prune_pi(W["store"], model, cs, folds, p.get("rounds"), int(p["seed"]), int(p.get("repeats", 3)))
        return {"pruned_ext": list(pruned.ext), "pi": df.to_dict("records"), "columns": list(df.columns)}
    if kind == "autots_bakeoff":
        from .cli import autots_bakeoff_fold

        cs, fold = _colset(p["colset"]), _fold(W, p["fold"])
        key = json.dumps(p["colset"], sort_keys=True)
        if key not in W["grid"]:
            W["grid"] = {key: W["store"].grid_matrix(cs)}  # cache 1 colset (regressor theo phút, immutable)
        name, params, all_t = autots_bakeoff_fold(W["cfg"], W["store"], fold, cs, p["group"], p["specs"], int(p["nv"]),
                                                  W["allow_cpu"], W["grid"][key])
        try:
            table = all_t.to_dict("records")
        except Exception:
            table = None
        return {"model": name, "params": params, "table": table}
    if kind == "autots_score":
        from .cli import _autots_probe_model
        from .harness import run_config

        m = _autots_probe_model(W["cfg"], p["group"], W["allow_cpu"], frozen=tuple(p["frozen"]))
        fold, cs = _fold(W, p["fold"]), _colset(p["colset"])
        r = run_config(W["store"], m, cs, [fold], rounds=None, seed=int(p["seed"]), keep_states=bool(p.get("want_preds")))
        preds = (np.asarray(r.states[0].idx_val), np.asarray(r.states[0].yhat)) if p.get("want_preds") else None
        return {"rmse": r.rmse[0], "e0": r.e0[0], "preds": preds}
    raise KeyError(f"scheduler: task kind không hợp lệ: {kind}")


# ----------------------------------------------------------------------------- scheduler (parent)
class GpuScheduler:
    """N worker process = len(gpu_devices) × gpu_slots_per_device, mỗi worker khoá vào MỘT GPU vật lý.

    Điều phối: task sẵn sàng → GPU rảnh (round-robin giữa các branch đang hoạt động, FIFO trong từng branch)
    → không branch nào chiếm hết máy, không starvation, không affinity theo model family.
    """

    def __init__(self, cfg, allow_cpu: bool = False, exp_dir: Path | None = None, light: bool = False):
        self.cfg = cfg
        self.allow_cpu = bool(allow_cpu)
        self.light = bool(light)  # worker không load data (chỉ kiểm thiết bị)
        self.devices, self.slots, self.n_workers = gpu.worker_slots(cfg)
        self.exp_dir = Path(exp_dir) if exp_dir is not None else Path(getattr(cfg, "exp_dir", "."))
        self.require_gpu = bool(getattr(cfg, "require_gpu", True))
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._pending: dict[str, deque] = {}
        self._branch_order: list[str] = []
        self._rr = 0
        self._groups: dict[str, tuple[_Group, int]] = {}
        self._idle: list[int] = []
        self._busy: dict[int, str] = {}
        self._procs: dict[int, Any] = {}
        self._queues: dict[int, Any] = {}
        self._reports: dict[int, dict] = {}
        self._dead: dict[int, str] = {}
        self._seq = 0
        self._dispatched: dict[str, tuple] = {}
        self._started = False
        self._stopping = False
        self._log_lock = threading.Lock()

    # ---------------------------------------------------------------- vòng đời
    def start(self) -> "GpuScheduler":
        if self._started:
            return self
        with self._cv:  # khởi động lại sau shutdown: dọn sạch trạng thái worker cũ
            self._stopping = False
            self._reports, self._dead, self._idle, self._busy, self._dispatched = {}, {}, [], {}, {}
        ctx = get_context("spawn")  # spawn: parent có thể đã init CUDA (gpu preflight) → cấm fork
        self._result_q = ctx.Queue()
        # Precompute ext của S0_m + Candidate_m ĐÚNG MỘT LẦN, TRONG PROCESS CHA, TRƯỚC KHI spawn worker nào (§ run
        # ML+LSTM): trước đây mỗi worker tự `load_store` + tự tính lại → cùng cột bị compute_short/compute_ext
        # NHIỀU LẦN (một lần mỗi worker). Giờ chỉ tính một lần ở đây rồi truyền dict kết quả cho MỌI worker; mỗi
        # worker chỉ nạp lại (`_worker_store`), không tự gọi compute_short/compute_ext nữa. Lỗi ở đây (nếu có) nổi
        # lên NGAY tại đây — chưa worker nào được spawn (vòng for dưới chưa chạy) — không fallback on-demand.
        precomputed_ext = {} if self.light else _precompute_ext(self.cfg)
        for wid in range(self.n_workers):
            dev = gpu.device_for_worker(wid, self.devices)
            q = ctx.Queue()
            p = ctx.Process(target=_worker_main,
                            args=(wid, dev, q, self._result_q, self.cfg, self.allow_cpu, self.require_gpu, self.light, precomputed_ext),
                            daemon=True, name=f"p0-gpu{dev}-w{wid}")
            p.start()
            self._queues[wid], self._procs[wid] = q, p
        self._started = True
        self._collector = threading.Thread(target=self._collect_loop, name="p0-sched-collect", daemon=True)
        self._collector.start()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, name="p0-sched-dispatch", daemon=True)
        self._dispatcher.start()
        self._await_ready()
        if self.n_workers > len(self.devices) * self.slots:
            self._record_warn("GPU_OVERSUBSCRIBE",
                              f"P0_FOLD_WORKERS={self.n_workers} > {len(self.devices)}×{self.slots} slot GPU — oversubscribe do user chỉ định")
        lgbm_dev = str((getattr(self.cfg, "models", {}) or {}).get("lgbm", {}).get("device_type", ""))
        if len(self.devices) > 1 and lgbm_dev == "gpu":
            # device_type=gpu = build OpenCL. Driver NVIDIA áp CUDA_VISIBLE_DEVICES cho cả OpenCL, nhưng đây là hành vi
            # của driver chứ không phải tham số của LightGBM → ghi WARN để `gpu-probe`/nvidia-smi xác nhận trên máy thật.
            self._record_warn("LGBM_OPENCL_ROUTING",
                              f"models.lgbm.device_type='gpu' (build OpenCL) trên {len(self.devices)} GPU: định tuyến dựa vào driver NVIDIA "
                              "áp CUDA_VISIBLE_DEVICES cho OpenCL. Xác nhận bằng `nvidia-smi` lúc chạy (mỗi worker chỉ nạp GPU của mình); "
                              "nếu không đúng, build LightGBM CUDA (device_type='cuda') hoặc chạy 1 GPU (P0_GPU_DEVICES=0).")
        return self

    def _await_ready(self, timeout: float | None = None) -> None:
        # worker phải đọc data + dựng feature B0 (data 2 năm ≈ 1,05 M bar) trước khi sẵn sàng → chờ rộng tay;
        # chỉnh bằng P0_WORKER_READY_TIMEOUT nếu máy chậm.
        timeout = float(timeout or os.environ.get("P0_WORKER_READY_TIMEOUT", 1800.0))
        t0 = time.time()
        with self._cv:
            while len(self._reports) + len(self._dead) < self.n_workers:
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"scheduler: worker không khởi động sau {timeout}s ({len(self._reports)}/{self.n_workers} sẵn sàng)")
                self._cv.wait(1.0)
            if self._dead:
                bad = "; ".join(f"worker {w}: {e}" for w, e in self._dead.items())
                self.shutdown()
                # §10: sự cố TÀI NGUYÊN GPU → dừng an toàn và HỎI USER (CLI biến thành `checker_log.gpu_stop`)
                raise gpu.GpuResourceError(f"worker GPU không khởi động được (không có CPU fallback) — {bad}")

    def shutdown(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        for wid, q in self._queues.items():
            try:
                q.put(None)
            except Exception:
                pass
        for wid, p in self._procs.items():
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._procs, self._queues = {}, {}
        self._started = False

    # ---------------------------------------------------------------- submit
    def submit(self, tasks: list[Task], branch: str | None = None) -> list[Any]:
        """Chạy một NHÓM task song song trên các GPU rảnh; trả kết quả theo ĐÚNG thứ tự `tasks` (tất định)."""
        if not tasks:
            return []
        if not self._started:
            self.start()
        br = branch or current_branch()
        ctx = current_ctx()
        g = _Group(tasks, [None] * len(tasks), [None] * len(tasks), len(tasks))
        with self._cv:
            for i, t in enumerate(tasks):
                self._seq += 1
                t.branch = br
                t.task_id = f"t{self._seq:06d}"
                t.stage = t.stage or str(ctx.get("stage", ""))
                t.candidate = t.candidate or str(ctx.get("candidate", ""))
                self._groups[t.task_id] = (g, i)
                self._pending.setdefault(br, deque()).append(t)
            if br not in self._branch_order:
                self._branch_order.append(br)
            self._cv.notify_all()
        g.done.wait()
        errs = [e for e in g.errors if e]
        if errs:
            msg = f"{len(errs)} task lỗi trên GPU (không có CPU fallback):\n" + "\n".join(errs[:3])
            if any(gpu.is_gpu_failure(e) for e in errs):  # §10: lỗi TÀI NGUYÊN GPU → dừng an toàn + hỏi user
                raise gpu.GpuResourceError(msg)
            raise RuntimeError("scheduler: " + msg)  # lỗi khác (bug/khoa học) → dừng tự động như cũ
        return g.results

    # ---------------------------------------------------------------- vòng điều phối
    def _next_task(self) -> Task | None:
        """Round-robin giữa các branch có task chờ (FIFO trong branch) → không branch nào độc chiếm máy (§11)."""
        names = [b for b in self._branch_order if self._pending.get(b)]
        if not names:
            return None
        self._rr = self._rr % len(names)
        b = names[self._rr]
        self._rr = (self._rr + 1) % len(names)
        return self._pending[b].popleft()

    def _dispatch_loop(self) -> None:
        while True:
            with self._cv:
                while not self._stopping and (not self._idle or not any(self._pending.values())):
                    self._cv.wait(0.5)
                    if self._stopping:
                        return
                if self._stopping:
                    return
                wid = self._idle.pop(0)
                t = self._next_task()
                if t is None:
                    self._idle.insert(0, wid)
                    continue
                self._busy[wid] = t.task_id
                self._dispatched[t.task_id] = (t, wid, time.time())
            self._queues[wid].put({"task_id": t.task_id, "kind": t.kind, "payload": t.payload})

    def _collect_loop(self) -> None:
        while True:
            try:
                msg = self._result_q.get(timeout=1.0)
            except (queue.Empty, OSError, ValueError):
                if self._stopping:
                    return
                self._check_dead_workers()
                continue
            with self._cv:
                if msg["type"] == "ready":
                    self._reports[msg["worker_id"]] = msg["report"]
                    self._idle.append(msg["worker_id"])
                    self._cv.notify_all()
                    continue
                if msg["type"] == "init_error":
                    self._dead[msg["worker_id"]] = msg["error"]
                    self._cv.notify_all()
                    continue
                self._finish(msg)

    def _finish(self, msg: dict) -> None:
        """(giữ lock) ghi kết quả về group + scheduler_log + trả worker về idle."""
        tid = msg["task_id"]
        t, wid, disp_at = self._dispatched.pop(tid, (None, msg.get("worker_id"), None))
        g_i = self._groups.pop(tid, None)
        self._busy.pop(msg["worker_id"], None)
        if msg["worker_id"] not in self._idle and msg["worker_id"] not in self._dead:
            self._idle.append(msg["worker_id"])
        self._log_task(t, msg, disp_at)
        if g_i is None:
            return
        g, i = g_i
        g.results[i] = msg["value"]
        g.errors[i] = msg["error"]
        g.remaining -= 1
        if g.remaining == 0:
            g.done.set()
        self._cv.notify_all()

    def _check_dead_workers(self) -> None:
        with self._cv:
            for wid, p in list(self._procs.items()):
                if p.is_alive() or wid in self._dead:
                    continue
                self._dead[wid] = (f"worker process trên GPU vật lý {gpu.device_for_worker(wid, self.devices)} chết "
                                   f"(exitcode={p.exitcode}) — nghi hết VRAM/OOM hoặc driver; KHÔNG có CPU fallback")
                if wid in self._idle:
                    self._idle.remove(wid)
                tid = self._busy.pop(wid, None)
                if tid is not None:
                    self._finish({"type": "result", "worker_id": wid, "device": gpu.device_for_worker(wid, self.devices), "task_id": tid,
                                  "value": None, "error": self._dead[wid], "started_at": time.time(), "ended_at": time.time(),
                                  "peak_vram_mb": None})
                if not self._idle and not any(p2.is_alive() for p2 in self._procs.values()):
                    for tid2, (g, i) in list(self._groups.items()):
                        g.errors[i] = self._dead[wid]
                        g.remaining -= 1
                        self._groups.pop(tid2, None)
                        if g.remaining == 0:
                            g.done.set()

    # ---------------------------------------------------------------- log (§20)
    def _log_task(self, t: Task | None, msg: dict, disp_at: float | None) -> None:
        row = {"timestamp_start": _iso(msg.get("started_at")), "timestamp_end": _iso(msg.get("ended_at")),
               "task_id": msg.get("task_id"), "kind": (t.kind if t else ""), "stage": (t.stage if t else ""),
               "branch": (t.branch if t else ""), "model": (t.model if t else ""), "fold": (t.fold if t else ""),
               "seed": (t.seed if t else None), "candidate": (t.candidate if t else ""),
               "gpu_physical_id": msg.get("device"), "worker_id": msg.get("worker_id"),
               "status": "error" if msg.get("error") else "ok",
               "t_start": round(float(msg.get("started_at") or 0.0), 6), "t_end": round(float(msg.get("ended_at") or 0.0), 6),
               "duration_sec": round(float(msg.get("ended_at", 0)) - float(msg.get("started_at", 0)), 3),
               "queue_wait_sec": (round(float(msg.get("started_at", 0)) - disp_at, 3) if disp_at else None),
               "peak_vram_mb": (round(msg["peak_vram_mb"], 1) if msg.get("peak_vram_mb") else None),
               "error": (str(msg.get("error"))[:400] if msg.get("error") else None)}
        self._append_log(row)

    def _append_log(self, row: dict) -> None:
        path = self.exp_dir / SCHED_LOG
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _record_warn(self, check_id: str, message: str) -> None:
        from .checker_log import record

        record(self.exp_dir, "scheduler", "WARN", check_id, message)

    # ---------------------------------------------------------------- thông tin
    @property
    def worker_devices(self) -> dict[int, int]:
        return {wid: gpu.device_for_worker(wid, self.devices) for wid in range(self.n_workers)}

    def reports(self) -> dict[int, dict]:
        return dict(self._reports)


def _iso(ts) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(float(ts))) if ts else None


def read_scheduler_log(exp_dir: Path) -> list[dict]:
    p = Path(exp_dir) / SCHED_LOG
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def in_worker() -> bool:
    """True nếu code đang chạy BÊN TRONG worker GPU (không được đệ quy vào scheduler nữa)."""
    return os.environ.get(gpu.ENV_WORKER_ID) is not None
