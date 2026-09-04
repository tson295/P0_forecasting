"""Scheduler 2 GPU đối xứng + DAG nhánh + champion replay (pass thực thi 2026-09-04c).

Khoá đúng những gì user yêu cầu — CHỈ thực thi, không đổi khoa học:
(9) hai worker GPU đối xứng · (10) không affinity model-family → GPU · (11)(12) 5 fold rải động, GPU rảnh nhận task kế
tiếp · (13) candidate j+1 chỉ bắt đầu khi candidate j xong hết fold + quyết định · (14) nhánh độc lập chạy đồng thời ·
(15) autots-search chỉ sẵn sàng khi có CẢ hai probe winner · (16) tfm-final chỉ sau khi nhánh TimesFM xong ·
(17) thứ tự chạy xong KHÔNG đổi thứ tự champion · (18) champion replay không train/inference · (19) ghép kết quả theo
thứ tự fold chuẩn · (20) kết quả y hệt bản tuần tự · (21) một GPU vẫn chạy · (22) gán thiết bị tường minh ·
(23) không CPU fallback · (24) artifact race-safe · (25) task dài ngắn khác nhau vẫn giữ cả hai GPU bận.

Chạy trên CPU với data tổng hợp (`allow_cpu`, `dataset_label = synthetic*`): scheduler/DAG là cơ chế thực thi nên
kiểm được không cần GPU thật; định tuyến GPU vật lý được kiểm qua CUDA_VISIBLE_DEVICES của từng worker.
"""
import json
import threading
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from p0 import cli, fold_parallel, gpu, scheduler
from p0.checker_log import read as read_checker
from p0.config import RunConfig
from p0.harness import ColSet, run_config
from p0.orchestrate import Branch, build_dag, run_dag
from p0.scheduler import GpuScheduler, Task
from p0.synthetic import make_hf, make_lf


# ----------------------------------------------------------------------------- fixtures
def _cfg(tmp_path, days=10.0, **over):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    hf = make_hf(n_days=days, seed=5)
    hf.to_csv(tmp_path / "data" / "hf.csv", index=False)
    make_lf(hf).to_csv(tmp_path / "data" / "lf.csv", index=False)
    (tmp_path / ".claude").mkdir(exist_ok=True)
    (tmp_path / ".claude" / "MEMORY.md").write_text("TRAINING: UNLOCKED\n", encoding="utf-8")
    cfg = RunConfig(dataset_label="synthetic_sched", hf_csv="data/hf.csv", lf_csv="data/lf.csv",
                    val_days=["2026-01-08", "2026-01-09"], test_start="2026-01-10", root=str(tmp_path), require_gpu=False,
                    calib_seed=1, eval_seeds=(2, 3), selection_seed=2,
                    **{"models": {"lgbm": {"n_jobs": 1, "n_estimators": 12, "min_child_samples": 20}}, **over})
    cli.cmd_check_data(cfg, Namespace(write_checksums=True))
    return cfg


@pytest.fixture
def two_gpu(monkeypatch):
    monkeypatch.setenv(gpu.ENV_DEVICES, "0,1")
    monkeypatch.delenv(gpu.ENV_WORKERS, raising=False)
    monkeypatch.delenv(gpu.ENV_SLOTS, raising=False)
    yield
    fold_parallel.shutdown()


def _sched(cfg, light=True):
    return GpuScheduler(cfg, allow_cpu=True, exp_dir=cfg.exp_dir, light=light).start()


def _log(cfg):
    return scheduler.read_scheduler_log(cfg.exp_dir)


# ----------------------------------------------------------------------------- (9)(22) worker đối xứng, thiết bị tường minh
def test_two_workers_are_symmetric_with_explicit_device_binding(tmp_path, two_gpu):
    cfg = _cfg(tmp_path)
    devices, slots, n = gpu.worker_slots(cfg)
    assert devices == [0, 1] and slots == 1 and n == 2  # mặc định: 1 task nặng / 1 GPU vật lý
    sch = _sched(cfg)
    try:
        reps = sch.reports()
        assert sorted(r["gpu_physical_id"] for r in reps.values()) == [0, 1]
        assert {r["cuda_visible_devices"] for r in reps.values()} == {"0", "1"}  # (22) mỗi worker chỉ thấy GPU của mình
        assert sch.worker_devices == {0: 0, 1: 1}
        out = sch.submit([Task(kind="probe", payload={"sleep_ms": 60, "tag": f"t{i}"}) for i in range(4)])
        assert {r["gpu_physical_id"] for r in out} == {0, 1}  # cả hai GPU đều nhận việc
    finally:
        sch.shutdown()


def test_worker_device_map_ignores_model_family(tmp_path, two_gpu):
    """(10) không có bảng model→GPU: `device_for_worker` chỉ phụ thuộc worker id; mỗi model đều dùng CẢ hai GPU."""
    assert gpu.device_for_worker(0, [0, 1]) == 0 and gpu.device_for_worker(1, [0, 1]) == 1
    assert gpu.device_for_worker(2, [0, 1]) == 0  # vòng tròn, không theo family
    # Đường ĐIỀU PHỐI (chọn worker/thiết bị) không được nhìn thấy model: kiểm bằng source của chính các hàm đó
    import inspect

    for fn in (gpu.device_for_worker, gpu.bind_worker_device, GpuScheduler._next_task, GpuScheduler._dispatch_loop):
        src = inspect.getsource(fn)
        if fn.__doc__:
            src = src.replace(fn.__doc__, "")
        body = " ".join(ln.split("#")[0] for ln in src.splitlines())  # bỏ docstring + comment, chỉ còn code
        assert "model" not in body and "family" not in body, f"{fn.__name__} không được phụ thuộc model/family"
    for family in ("lgbm", "xgb", "cat", "lstm", "tfm", "autots"):  # không có bảng model → GPU ở bất kỳ đâu
        assert f'"{family}": ' not in Path("src/p0/scheduler.py").read_text(encoding="utf-8")
        assert f'"{family}"' not in Path("src/p0/gpu.py").read_text(encoding="utf-8")
    cfg = _cfg(tmp_path)
    sch = _sched(cfg)
    try:
        used = {}
        for fam in ("tree_family", "dl_family"):
            out = sch.submit([Task(kind="probe", model=fam, payload={"sleep_ms": 120, "tag": f"{fam}{i}"}) for i in range(4)])
            used[fam] = {r["gpu_physical_id"] for r in out}
        assert used["tree_family"] == used["dl_family"] == {0, 1}  # ML và DL đều chạy trên cả hai GPU
    finally:
        sch.shutdown()


# ----------------------------------------------------------------------------- (11)(12)(19)(20) fold rải động
def test_folds_spread_dynamically_and_results_match_sequential(tmp_path, two_gpu):
    cfg = _cfg(tmp_path)
    store, folds, _, _ = cli.load_store(cfg)
    model = cli.model_for(cfg, "lgbm", allow_cpu=True)
    cs = ColSet(store.b0_names[:12], ("ret_60",))
    rounds = {f.name: (6, 6, 6) for f in folds}
    seq = run_config(store, model, cs, folds, rounds=rounds, seed=3, keep_states=False)
    assert fold_parallel.configure(cfg, model, "lgbm", True) == 2 and fold_parallel.active(model)
    try:
        par = run_config(store, model, cs, folds, rounds=rounds, seed=3, keep_states=False)
    finally:
        fold_parallel.shutdown()
    # (19) ghép theo ĐÚNG thứ tự fold chuẩn; (20) cùng seed → cùng số
    assert par.fold_names == seq.fold_names == [f.name for f in folds]
    assert np.allclose(par.rmse, seq.rmse, rtol=1e-6, atol=1e-6) and np.allclose(par.e0, seq.e0)
    assert (par.best_iters == seq.best_iters).all() and par.rounds == seq.rounds
    rows = [r for r in _log(cfg) if r["kind"] == "run_fold"]
    assert len(rows) == len(folds) and {r["status"] for r in rows} == {"ok"}
    assert {r["gpu_physical_id"] for r in rows} == {0, 1}  # (11) 5 fold của MỘT cấu hình rải trên cả hai GPU
    assert all(r["model"] == "lgbm" and r["fold"] and r["seed"] == 3 for r in rows)


def test_free_gpu_takes_next_task_even_if_other_gpu_started_the_batch(tmp_path, two_gpu):
    """(12) GPU nào xong trước thì nhận task kế tiếp — không giữ chỗ theo nhánh/model đã chạy trước đó."""
    cfg = _cfg(tmp_path)
    sch = _sched(cfg)
    try:
        out = sch.submit([Task(kind="probe", payload={"sleep_ms": ms, "tag": f"t{i}"})
                          for i, ms in enumerate([900, 60, 60, 60, 60])])
        by_gpu = {}
        for r in out:
            by_gpu.setdefault(r["gpu_physical_id"], 0)
            by_gpu[r["gpu_physical_id"]] += 1
        assert set(by_gpu) == {0, 1} and max(by_gpu.values()) >= 4  # GPU rảnh gánh phần lớn task ngắn
    finally:
        sch.shutdown()


def test_uneven_durations_keep_both_workers_busy(tmp_path, two_gpu):
    """(25) khi có ≥ 2 task sẵn sàng, hai GPU chạy CHỒNG thời gian (không tuần tự hoá)."""
    cfg = _cfg(tmp_path)
    sch = _sched(cfg)
    try:
        ms = [500, 120, 120, 500, 120, 120]
        t0 = time.time()
        sch.submit([Task(kind="probe", payload={"sleep_ms": m, "tag": f"u{i}"}) for i, m in enumerate(ms)])
        wall = time.time() - t0
    finally:
        sch.shutdown()
    rows = [r for r in _log(cfg) if r["kind"] == "probe"]
    assert wall < sum(ms) / 1000.0 * 0.8  # nhanh hơn hẳn tuần tự
    iv = sorted((r["t_start"], r["t_end"], r["gpu_physical_id"]) for r in rows)
    overlap = any(a[1] > b[0] and a[2] != b[2] for a, b in zip(iv, iv[1:]))
    assert overlap  # có ít nhất một khoảng hai GPU cùng chạy


# ----------------------------------------------------------------------------- (13) candidate tuần tự
def test_candidate_j_plus_1_starts_only_after_candidate_j_finished(tmp_path, two_gpu, monkeypatch):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    (exp / "s0").mkdir(parents=True, exist_ok=True)
    store, folds, _, _ = cli.load_store(cfg)
    ColSet(store.b0_names[:8]).save(exp / "s0" / "lgbm.json")
    (exp / "s0" / "candidates_lgbm.json").write_text(json.dumps(
        {"model": "lgbm", "candidates": ["ret_2", "ret_3", "rsi3_centered"], "audit_dataset_label": cfg.dataset_label}), encoding="utf-8")
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    try:
        cli.cmd_loop(cfg, Namespace(model="lgbm", smoke=True, allow_cpu=True, max_candidates=None, no_standalone=True,
                                    latency_origins=None, resume=False))
    finally:
        fold_parallel.shutdown()
    rows = [r for r in _log(cfg) if r["stage"] == "add_one"]
    assert rows, "phải có task add-one đi qua scheduler"
    span = {}
    for r in rows:
        lo, hi = span.get(r["candidate"], (r["t_start"], r["t_end"]))
        span[r["candidate"]] = (min(lo, r["t_start"]), max(hi, r["t_end"]))
    order = sorted(span.items(), key=lambda kv: kv[1][0])
    assert [c for c, _ in order] == ["ret_2", "ret_3", "rsi3_centered"]
    for (_, a), (_, b) in zip(order, order[1:]):
        assert a[1] <= b[0] + 1e-6  # candidate sau chỉ bắt đầu khi candidate trước đã xong TOÀN BỘ fold
    # prune PI chạy trọn vẹn trong MỘT worker (một dòng RNG như bản tuần tự)
    assert len([r for r in _log(cfg) if r["kind"] == "prune_pi"]) == 1


# ----------------------------------------------------------------------------- (14)(15)(16) DAG nhánh
def _stub_dag_run(order_log, lock, durations):
    def run_branch(b: Branch):
        with lock:
            order_log.append(("start", b.name, time.time()))
        time.sleep(durations.get(b.name, 0.05))
        with lock:
            order_log.append(("end", b.name, time.time()))
    return run_branch


def test_independent_branches_run_concurrently_and_deps_are_respected():
    """(14) autots_wr/autots_mr chạy đồng thời; (15) autots-search chỉ sau CẢ hai; (16) tfm-final chỉ sau loop tfm."""
    branches = build_dag(["tfm", "autots_wr", "autots_mr"])
    names = [b.name for b in branches]
    assert names == ["loop:tfm", "loop:autots_wr", "loop:autots_mr", "tfm-final", "autots-search"]
    assert dict((b.name, b.deps) for b in branches)["autots-search"] == ("loop:autots_wr", "loop:autots_mr")
    assert dict((b.name, b.deps) for b in branches)["tfm-final"] == ("loop:tfm",)
    ev, lock = [], threading.Lock()
    run_dag(branches, _stub_dag_run(ev, lock, {"loop:tfm": 0.3, "loop:autots_wr": 0.3, "loop:autots_mr": 0.3}), max_active=2)
    t = {(k, n): ts for k, n, ts in ev}
    assert t[("start", "autots-search")] >= t[("end", "loop:autots_wr")] - 1e-6
    assert t[("start", "autots-search")] >= t[("end", "loop:autots_mr")] - 1e-6
    assert t[("start", "tfm-final")] >= t[("end", "loop:tfm")] - 1e-6
    # hai probe AutoTS chồng thời gian với nhau hoặc với loop:tfm — không nhánh nào phải chờ nhánh độc lập
    running = [(t[("start", n)], t[("end", n)]) for n in ("loop:tfm", "loop:autots_wr", "loop:autots_mr")]
    running.sort()
    assert any(a[1] > b[0] for a, b in zip(running, running[1:]))
    assert all(b.status == "done" for b in branches)


def test_dag_stops_starting_new_branches_after_failure():
    branches = build_dag(["lgbm", "xgb", "tfm"])

    def boom(b: Branch):
        if b.name == "loop:lgbm":
            raise RuntimeError("OOM giả lập trên GPU 0")
        time.sleep(0.05)

    with pytest.raises(RuntimeError, match="OOM"):
        run_dag(branches, boom, max_active=2)
    assert dict((b.name, b.status) for b in branches)["tfm-final"] == "pending"  # dep chưa xong → không khởi động


# ----------------------------------------------------------------------------- (17)(18) champion replay
def _win_file(exp: Path, model: str, rmse, eps=0.02, e0=(100.0, 140.0, 170.0)):
    (exp / "wins").mkdir(parents=True, exist_ok=True)
    (exp / "wins" / f"{model}.json").write_text(json.dumps(
        {"model": model, "colset": {"b0": [], "ext": []}, "rmse_mean": [list(rmse)], "e0": [list(e0)], "eps": eps,
         "eval_seeds": [2, 3], "which": "prune", "median_gain_vs_e0": 0.5,
         "champion_extra": {"win": "prune", "train_device": "GPU"}}), encoding="utf-8")


def _replay_cfg(tmp_path, root_name):
    root = tmp_path / root_name
    (root / ".claude").mkdir(parents=True, exist_ok=True)
    (root / ".claude" / "MEMORY.md").write_text("TRAINING: LOCKED\n", encoding="utf-8")
    return RunConfig(dataset_label="synthetic_replay", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"],
                     test_start="2026-01-04", root=str(root), require_gpu=False, defer_champion=True,
                     model_order=["lgbm", "xgb", "cat", "tfm", "xgbrf", "autots_wr", "autots_mr", "lstm"])


RMSE = {"lgbm": (95.0, 135.0, 165.0), "xgb": (94.0, 134.0, 164.0), "cat": (96.0, 136.0, 166.0), "tfm": (99.0, 139.0, 169.0),
        "xgbrf": (93.0, 133.0, 163.0), "autots": (98.0, 138.0, 168.0), "lstm": (97.0, 137.0, 167.0)}


def test_champion_replay_order_is_fixed_regardless_of_finish_order(tmp_path):
    """(17) thứ tự HOÀN THÀNH của nhánh không đổi thứ tự so champion (methodology cố định)."""
    import pandas as pd

    champs = []
    for name, write_order in (("a", list(RMSE)), ("b", list(reversed(list(RMSE))))):
        cfg = _replay_cfg(tmp_path, name)
        exp = cfg.exp_dir
        for m in write_order:  # "chạy xong" theo thứ tự khác nhau
            _win_file(exp, m, RMSE[m])
            time.sleep(0.001)
        cli.cmd_champion_replay(cfg, Namespace(allow_partial=False, force_replay=False))
        ch = pd.read_csv(exp / "champion_log.csv")
        assert list(ch["model"]) == list(cli.CHAMPION_ORDER)  # thứ tự cố định lgbm→xgb→cat→tfm→xgbrf→autots→lstm
        rep = json.loads((exp / "champion_replay.json").read_text(encoding="utf-8"))
        assert rep["order"] == list(cli.CHAMPION_ORDER) and rep["missing"] == []
        champs.append(json.loads((exp / "champion.json").read_text(encoding="utf-8"))["model"])
    assert champs[0] == champs[1] == "xgbrf"  # cùng champion dù thứ tự chạy xong đảo ngược


def test_champion_replay_runs_no_training_or_inference(tmp_path, monkeypatch):
    """(18) replay chỉ đọc artifact: không load data, không train, không inference."""
    cfg = _replay_cfg(tmp_path, "c")
    exp = cfg.exp_dir
    for m in RMSE:
        _win_file(exp, m, RMSE[m])

    def boom(*a, **k):
        raise AssertionError("champion replay không được train/inference")

    monkeypatch.setattr(cli, "load_store", boom)
    monkeypatch.setattr(cli, "run_config", boom)
    monkeypatch.setattr(cli, "model_for", boom)
    monkeypatch.setattr(cli, "gate", boom)
    cli.cmd_champion_replay(cfg, Namespace(allow_partial=False, force_replay=False))
    assert json.loads((exp / "champion.json").read_text(encoding="utf-8"))["model"] == "xgbrf"


def test_champion_replay_requires_all_representatives_and_lgbm_first(tmp_path):
    cfg = _replay_cfg(tmp_path, "d")
    exp = cfg.exp_dir
    for m in ("lgbm", "xgb"):
        _win_file(exp, m, RMSE[m])
    with pytest.raises(SystemExit, match="REPRESENTATIVE_MISSING"):
        cli.cmd_champion_replay(cfg, Namespace(allow_partial=False, force_replay=False))
    cli.cmd_champion_replay(cfg, Namespace(allow_partial=True, force_replay=False))  # cố ý replay một phần
    assert json.loads((exp / "champion.json").read_text(encoding="utf-8"))["model"] == "xgb"
    with pytest.raises(SystemExit, match="CHAMPION_EXISTS"):  # replay là bước DUY NHẤT ghi champion state
        cli.cmd_champion_replay(cfg, Namespace(allow_partial=True, force_replay=False))


def test_probe_and_internal_configs_can_never_touch_champion(tmp_path):
    """(7)(8) chỉ đại diện mới được so champion — probe/cấu hình nội bộ bị chặn cứng."""
    cfg = _replay_cfg(tmp_path, "e")
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    for bad in ("tfm_lora_baseline", "tfm_lora_xreg", "autots_wr", "autots_mr"):
        with pytest.raises(SystemExit, match="CHAMPION_INELIGIBLE"):
            cli.champion_step(cfg, bad, ColSet((), ()), np.array([[90.0, 130.0, 160.0]]), np.array([[100.0, 140.0, 170.0]]), 0.02)


# ----------------------------------------------------------------------------- (21)(23) một GPU / không CPU fallback
def test_single_gpu_mode_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv(gpu.ENV_DEVICES, "0")
    monkeypatch.delenv(gpu.ENV_WORKERS, raising=False)
    cfg = _cfg(tmp_path)
    devices, slots, n = gpu.worker_slots(cfg)
    assert devices == [0] and n == 1
    store, folds, _, _ = cli.load_store(cfg)
    model = cli.model_for(cfg, "lgbm", allow_cpu=True)
    cs = ColSet(store.b0_names[:10])
    rounds = {f.name: (5, 5, 5) for f in folds}
    seq = run_config(store, model, cs, folds, rounds=rounds, seed=2, keep_states=False)
    assert fold_parallel.configure(cfg, model, "lgbm", True) == 1 and not fold_parallel.active(model)  # 1 GPU = chạy trong process cha
    sch = _sched(cfg, light=False)  # scheduler 1 worker vẫn hoạt động và cho cùng số
    try:
        assert sch.worker_devices == {0: 0}
        out = sch.submit([Task(kind="run_fold", model="lgbm", fold=f.name, seed=2,
                               payload={"model": "lgbm", "colset": cs.to_dict(), "fold": f.name, "rounds": rounds, "seed": 2,
                                        "want_yhat": False, "latency_origins": None}) for f in folds])
        assert np.allclose(np.array([r["rmse"] for r in out]), seq.rmse, atol=1e-6)
        assert {r["fold"] for r in out} == {f.name for f in folds}
    finally:
        sch.shutdown()


def test_no_cpu_fallback_anywhere(tmp_path, monkeypatch):
    """(23) thiếu GPU đúng như giao → lỗi rõ ràng, không bao giờ tự chuyển sang CPU."""
    monkeypatch.setenv(gpu.ENV_PHYSICAL, "1")
    monkeypatch.setenv(gpu.ENV_WORKER_ID, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    class _FakeCuda:
        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def is_available():
            return False

    import sys
    import types

    fake = types.SimpleNamespace(cuda=_FakeCuda)
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="không có CPU fallback"):
        gpu.device_report(require_gpu=True)
    assert gpu.device_report(require_gpu=False)["gpu_physical_id"] == 1
    # data/checksum sai → scheduler báo lỗi và dừng, không có nhánh "chạy CPU cho xong". Từ khi precompute ext
    # chạy TRONG PROCESS CHA trước khi spawn worker nào (§ run ML+LSTM), checksum mismatch nổi lên NGAY ở đó
    # (hard_fail SystemExit, KHÔNG bị nuốt) thay vì phải đợi một worker khởi động hỏng mới báo GpuResourceError.
    monkeypatch.delitem(sys.modules, "torch")
    bad = _cfg(tmp_path)
    (Path(bad.root) / "data" / "hf.csv").write_text("datetime,timestamp,open,high,low,close,volume,amount\n", encoding="utf-8")
    monkeypatch.setenv(gpu.ENV_DEVICES, "0,1")
    with pytest.raises(SystemExit, match="CHECKSUM_MISMATCH"):
        GpuScheduler(bad, allow_cpu=True, exp_dir=bad.exp_dir).start()


# ----------------------------------------------------------------------------- (24) artifact race-safe
def test_scheduler_and_log_artifacts_are_race_safe(tmp_path, two_gpu):
    import pandas as pd

    from p0.checker_log import read as read_checker
    from p0.checker_log import record as ck_record
    from p0.logs import log_run

    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir

    def spam(i):
        for k in range(25):
            log_run(exp, {"exp_id": f"e{i}_{k}", "step": "loop", "model": f"m{i}", "note": "x" * 50})
            ck_record(exp, "test", "INFO", "RACE", f"branch {i} finding {k}")

    threads = [threading.Thread(target=spam, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    df = pd.read_csv(exp / "log.csv")
    assert len(df) == 100 and df["exp_id"].nunique() == 100 and set(df["step"]) == {"loop"}
    assert len(read_checker(exp)) == 100  # mọi dòng JSONL hợp lệ
    sch = _sched(cfg)
    try:
        sch.submit([Task(kind="probe", payload={"sleep_ms": 20, "tag": f"r{i}"}) for i in range(8)])
    finally:
        sch.shutdown()
    rows = _log(cfg)
    assert len([r for r in rows if r["kind"] == "probe"]) == 8
    assert all(isinstance(r["task_id"], str) and r["status"] == "ok" for r in rows if r["kind"] == "probe")
    assert len({r["task_id"] for r in rows}) == len(rows)  # không dòng nào bị ghi đè/xen kẽ


# ----------------------------------------------------------------------------- schema scheduler_log (§20)
def test_scheduler_log_schema(tmp_path, two_gpu):
    cfg = _cfg(tmp_path)
    sch = _sched(cfg)
    try:
        sch.submit([Task(kind="probe", model="lgbm", fold="fold1", seed=7, candidate="ret_2", stage="add_one",
                         payload={"sleep_ms": 10, "tag": "s"})])
    finally:
        sch.shutdown()
    row = [r for r in _log(cfg) if r["kind"] == "probe"][0]
    for k in ("timestamp_start", "timestamp_end", "task_id", "stage", "model", "fold", "seed", "candidate",
              "gpu_physical_id", "worker_id", "status", "duration_sec", "peak_vram_mb", "error", "branch"):
        assert k in row
    assert row["model"] == "lgbm" and row["fold"] == "fold1" and row["seed"] == 7 and row["candidate"] == "ret_2"
    assert row["stage"] == "add_one" and row["duration_sec"] >= 0 and row["gpu_physical_id"] in (0, 1)


# ----------------------------------------------------------------------------- orchestrate end-to-end (DAG + replay)
def test_orchestrate_runs_branches_in_parallel_then_replays_champion(tmp_path, two_gpu):
    """Tích hợp: hai nhánh model độc lập chạy đồng thời trên 2 worker GPU, champion CHỈ được quyết ở replay
    (thứ tự cố định), TEST không bị chạm."""
    import pandas as pd

    from p0.orchestrate import cmd_orchestrate, summarize_utilization

    cfg = _cfg(tmp_path, defer_champion=True, model_order=["lgbm", "xgb"],
               models={"lgbm": {"n_jobs": 1, "n_estimators": 12, "min_child_samples": 20},
                       "xgb": {"n_estimators": 12, "max_depth": 3}})
    exp = cfg.exp_dir
    (exp / "s0").mkdir(parents=True, exist_ok=True)
    store, folds, _, _ = cli.load_store(cfg)
    for m in ("lgbm", "xgb"):
        ColSet(store.b0_names[:8]).save(exp / "s0" / f"{m}.json")
        (exp / "s0" / f"candidates_{m}.json").write_text(json.dumps(
            {"model": m, "candidates": ["ret_2", "ret_3"], "audit_dataset_label": cfg.dataset_label}), encoding="utf-8")
    args = Namespace(models=None, max_candidates=2, no_standalone=True, latency_origins=None, resume=False, max_branches=2,
                     skip_ensemble=False, allow_partial=False, force_replay=False, dry_run=False, smoke=True, allow_cpu=True,
                     config=None)
    try:
        cmd_orchestrate(cfg, args)
    finally:
        fold_parallel.shutdown()
    # champion chỉ được ghi ở replay, theo thứ tự cố định (không theo nhánh nào xong trước)
    ch = pd.read_csv(exp / "champion_log.csv")
    assert list(ch["model"]) == ["lgbm", "xgb"]
    rep = json.loads((exp / "champion_replay.json").read_text(encoding="utf-8"))
    assert rep["order"] == ["lgbm", "xgb"] and (exp / "champion_replay.csv").exists()
    assert json.loads((exp / "champion.json").read_text(encoding="utf-8"))["model"] in ("lgbm", "xgb")
    for m in ("lgbm", "xgb"):
        w = json.loads((exp / "wins" / f"{m}.json").read_text(encoding="utf-8"))
        assert w["representative"] == m and "champion_extra" in w
    # hai nhánh dùng cả hai GPU và có chồng thời gian (không tuần tự hoá vì lý do thực thi)
    rows = [r for r in _log(cfg) if r["status"] == "ok"]
    assert {r["branch"] for r in rows} >= {"loop:lgbm", "loop:xgb"}
    assert {r["gpu_physical_id"] for r in rows} == {0, 1}
    lg = [(r["t_start"], r["t_end"]) for r in rows if r["branch"] == "loop:lgbm"]
    xg = [(r["t_start"], r["t_end"]) for r in rows if r["branch"] == "loop:xgb"]
    assert any(a[0] < b[1] and b[0] < a[1] for a in lg for b in xg)  # có lúc hai nhánh chạy cùng lúc
    assert summarize_utilization(exp)["tasks"] == len(_log(cfg))
    # TEST không bị chạm
    assert not (exp / "final").exists()
    orch = [json.loads(l) for l in (exp / "orchestrate_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["branch"] for r in orch} == {"loop:lgbm", "loop:xgb"} and all(r["status"] != "error" for r in orch)


def test_orchestrate_requires_deferred_champion_and_dry_run_lists_dag(tmp_path, two_gpu, capsys):
    from p0.orchestrate import cmd_orchestrate

    cfg = _cfg(tmp_path, model_order=["lgbm", "tfm", "autots_wr", "autots_mr"])
    args = Namespace(models=None, max_candidates=1, no_standalone=True, latency_origins=None, resume=False, max_branches=2,
                     skip_ensemble=True, allow_partial=False, force_replay=False, dry_run=True, smoke=True, allow_cpu=True, config=None)
    cmd_orchestrate(cfg, args)  # dry-run: chỉ in DAG
    out = capsys.readouterr().out
    assert "loop:tfm" in out and "tfm-final" in out and "autots-search" in out
    args.dry_run = False
    with pytest.raises(SystemExit, match="defer_champion"):  # nhánh song song mà không hoãn champion → từ chối
        cmd_orchestrate(cfg, args)


# ----------------------------------------------------------------------------- (18)(19)(20) gpu-probe: UUID + backend trong worker
def _fake_probe_result(worker_id, device, uuid, backends):
    return {"report": {"worker_id": worker_id, "gpu_physical_id": device, "cuda_visible_devices": str(device),
                       "device_name": "RTX 5000 Ada", "device_uuid": uuid, "torch_device_count": 1},
            "backends": backends}


OK_BACKENDS = {b: {"status": "ok", "detail": "probe OK"} for b in ("torch", "xgboost", "lightgbm", "catboost", "jax", "timesfm")}


def _patch_probe_scheduler(monkeypatch, results):
    """Thay GpuScheduler trong cmd_gpu_probe bằng bản giả trả `results` cho task backend_probe."""
    calls = {"backend_probe": 0, "probe": 0, "tasks": []}

    class FakeSched:
        def __init__(self, cfg, allow_cpu=False, exp_dir=None, light=False):
            self.exp_dir = exp_dir

        def start(self):
            return self

        def submit(self, tasks, branch=None):
            calls["tasks"].append([t.kind for t in tasks])
            if tasks[0].kind == "backend_probe":
                calls["backend_probe"] += len(tasks)
                assert len(tasks) == len(results), "phải probe MỌI worker (mỗi GPU vật lý một task)"
                return list(results)
            calls["probe"] += len(tasks)
            return [{"gpu_physical_id": i % 2, "tag": t.payload.get("tag")} for i, t in enumerate(tasks)]

        def reports(self):
            return {r["report"]["worker_id"]: r["report"] for r in results}

        def shutdown(self):
            pass

    monkeypatch.setattr("p0.scheduler.GpuScheduler", FakeSched)
    return calls


def test_gpu_probe_requires_distinct_uuid_per_worker(tmp_path, two_gpu, monkeypatch):
    """(18) hai worker báo CÙNG UUID = định tuyến GPU sai → dừng an toàn + hỏi user (exit 3)."""
    cfg = _cfg(tmp_path)
    same = "uuid-AAA"
    _patch_probe_scheduler(monkeypatch, [_fake_probe_result(0, 0, same, OK_BACKENDS), _fake_probe_result(1, 1, same, OK_BACKENDS)])
    with pytest.raises(SystemExit) as e:
        cli.cmd_gpu_probe(cfg, Namespace(allow_cpu=True, backends=None))
    assert e.value.code == 3  # exit riêng: chờ user quyết
    rows = [r for r in read_checker(cfg.exp_dir) if r["check_id"] == "GPU_UUID_COLLISION"]
    assert rows and rows[-1]["severity"] == "ERROR" and rows[-1]["ref"] == "USER_DECISION_REQUIRED"
    assert json.loads((cfg.exp_dir / "gpu_probe.json").read_text(encoding="utf-8"))["verdict"] == "GPU_UUID_COLLISION"


def test_gpu_probe_passes_with_distinct_uuid_and_probes_every_gpu(tmp_path, two_gpu, monkeypatch):
    """(19)(20) mỗi GPU cấu hình được probe bằng MỘT task chạy trong worker đã mask; UUID phân biệt → PASS."""
    cfg = _cfg(tmp_path)
    calls = _patch_probe_scheduler(monkeypatch, [_fake_probe_result(0, 0, "uuid-A", OK_BACKENDS),
                                                 _fake_probe_result(1, 1, "uuid-B", OK_BACKENDS)])
    cli.cmd_gpu_probe(cfg, Namespace(allow_cpu=True, backends=None))
    assert calls["backend_probe"] == 2 and calls["tasks"][0] == ["backend_probe", "backend_probe"]
    saved = json.loads((cfg.exp_dir / "gpu_probe.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == "OK" and sorted(saved["uuids"].values()) == ["uuid-A", "uuid-B"]
    assert {w["report"]["gpu_physical_id"] for w in saved["workers"].values()} == {0, 1}
    ck = [r["check_id"] for r in read_checker(cfg.exp_dir)]
    assert "GPU_UUID_DISTINCT" in ck and "GPU_SYMMETRIC" in ck


def test_gpu_probe_stops_when_installed_backend_cannot_use_gpu(tmp_path, two_gpu, monkeypatch):
    """(21)(22) backend đã cài nhưng không chạy được GPU → KHÔNG fallback CPU, dừng an toàn + hỏi user."""
    cfg = _cfg(tmp_path)
    bad = {**OK_BACKENDS, "lightgbm": {"status": "fail", "detail": "LightGBMError: CUDA Tree Learner was not enabled"}}
    _patch_probe_scheduler(monkeypatch, [_fake_probe_result(0, 0, "uuid-A", OK_BACKENDS), _fake_probe_result(1, 1, "uuid-B", bad)])
    with pytest.raises(SystemExit) as e:
        cli.cmd_gpu_probe(cfg, Namespace(allow_cpu=True, backends=None))
    assert e.value.code == 3
    rows = [r for r in read_checker(cfg.exp_dir) if r["check_id"] == "BACKEND_GPU_FAILED"]
    assert rows and rows[-1]["ref"] == "USER_DECISION_REQUIRED" and "worker1/lightgbm" in rows[-1]["message"]
    assert json.loads((cfg.exp_dir / "gpu_probe.json").read_text(encoding="utf-8"))["verdict"] == "BACKEND_GPU_FAILED"


def test_missing_backend_is_env_warning_not_gpu_failure(tmp_path, two_gpu, monkeypatch):
    """Thư viện CHƯA CÀI = vấn đề môi trường (WARN, chạy tiếp), không phải sự cố tài nguyên GPU."""
    cfg = _cfg(tmp_path)
    miss = {**OK_BACKENDS, "timesfm": {"status": "missing", "detail": "ModuleNotFoundError: timesfm"}}
    _patch_probe_scheduler(monkeypatch, [_fake_probe_result(0, 0, "uuid-A", miss), _fake_probe_result(1, 1, "uuid-B", miss)])
    cli.cmd_gpu_probe(cfg, Namespace(allow_cpu=True, backends=None))  # KHÔNG raise
    sev = {r["check_id"]: r["severity"] for r in read_checker(cfg.exp_dir)}
    assert sev.get("BACKEND_MISSING") == "WARN" and sev.get("GPU_SYMMETRIC") == "PASS"


def test_backend_probe_task_runs_inside_masked_worker(tmp_path, two_gpu):
    """(20) task `backend_probe` thực thi TRONG worker: báo cáo mang đúng worker_id/GPU vật lý của process đó."""
    cfg = _cfg(tmp_path)
    sch = _sched(cfg)
    try:
        out = sch.submit([Task(kind="backend_probe", stage="gpu_probe", payload={"backends": ["torch"], "lgbm_device_type": "cpu"})
                          for _ in range(2)])
    finally:
        sch.shutdown()
    ids = sorted(r["report"]["gpu_physical_id"] for r in out)
    assert ids == [0, 1]  # mỗi task chạy ở một worker khác nhau, mỗi worker một GPU vật lý
    for r in out:
        assert str(r["report"]["cuda_visible_devices"]) == str(r["report"]["gpu_physical_id"])
        assert set(r["backends"]) == {"torch"}


# ----------------------------------------------------------------------------- (22)(23)(24) chính sách dừng/hỏi
def test_gpu_resource_failure_is_the_only_user_prompt(tmp_path, capsys):
    from p0.checker_log import gpu_stop, hard_fail
    from p0.checker_log import record as ck_record

    exp = tmp_path / "experiments"
    # (22) sự cố GPU: ERROR + ref USER_DECISION_REQUIRED + exit 3 + có phương án cho user
    with pytest.raises(SystemExit) as e:
        gpu_stop(exp, "loop", "GPU_RESOURCE_FAILURE", "GPU 1 biến mất giữa chừng", model="xgb")
    assert e.value.code == 3
    out = capsys.readouterr().out
    assert "CẦN USER QUYẾT" in out and "Bạn muốn xử lý thế nào?" in out and "KHÔNG có CPU fallback" in out
    rows = read_checker(exp)
    assert rows[-1]["severity"] == "ERROR" and rows[-1]["ref"] == "USER_DECISION_REQUIRED"
    # (23) finding thường: ghi rồi đi tiếp, KHÔNG hỏi
    ck_record(exp, "loop", "WARN", "UNUSUAL_GAIN", "gain 1.2 pp")
    ck_record(exp, "loop", "INFO", "NOTE", "ghi chú")
    out = capsys.readouterr().out
    assert "Bạn muốn xử lý" not in out
    # (24) vi phạm bất biến khoa học: dừng ngay, KHÔNG có tuỳ chọn "chạy tiếp"
    with pytest.raises(SystemExit) as e:
        hard_fail(exp, "load_store", "CHECKSUM_MISMATCH", "sha256 lệch")
    assert e.value.code != 3 and "CHECKSUM_MISMATCH" in str(e.value)
    out = capsys.readouterr().out
    assert "Bạn muốn xử lý" not in out and "chạy tiếp" not in out


def test_gpu_failure_classification():
    assert gpu.is_gpu_failure("CUDA out of memory") and gpu.is_gpu_failure("worker process chết (exitcode=1)")
    assert gpu.is_gpu_failure("GPU preflight XGBoost: booster báo device='cpu'")
    assert not gpu.is_gpu_failure("checksum data không khớp §6.1")
    assert not gpu.is_gpu_failure("locked_ext phải là tập con của ext")


def test_cli_main_turns_gpu_resource_error_into_user_decision_stop(tmp_path, monkeypatch, capsys):
    """(22) BẤT KỲ bước nào ném GpuResourceError → CLI dừng an toàn và hỏi user (exit 3), không CPU fallback."""
    cfg = _cfg(tmp_path)
    cfg_path = tmp_path / "configs" / "cli.json"
    cfg_path.parent.mkdir(exist_ok=True)
    d = {k: v for k, v in cfg.to_dict().items() if k != "root"}
    cfg_path.write_text(json.dumps(d), encoding="utf-8")

    def boom(c, a):
        raise gpu.GpuResourceError("worker GPU không khởi động được (không có CPU fallback) — worker 1: CUDA driver lỗi")

    monkeypatch.setattr(cli, "cmd_lock_s0", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["lock-s0", "--config", str(cfg_path)])
    assert e.value.code == 3
    out = capsys.readouterr().out
    assert "DỪNG AN TOÀN" in out and "CPU fallback" in out
    rows = [r for r in read_checker(cfg.exp_dir) if r["check_id"] == "GPU_RESOURCE_FAILURE"]
    assert rows and rows[-1]["ref"] == "USER_DECISION_REQUIRED"


# ----------------------------------------------------------------------------- (25)(26) đồng thời tối đa = số slot GPU
def _max_concurrency(rows):
    ev = []
    for r in rows:
        ev.append((r["t_start"], 1))
        ev.append((r["t_end"], -1))
    cur = best = 0
    for _, d in sorted(ev):
        cur += d
        best = max(best, cur)
    return best


def test_one_slot_per_gpu_means_at_most_two_heavy_tasks(tmp_path, two_gpu):
    """(25) gpu_slots_per_device=1 → không bao giờ quá 2 task nặng chạy cùng lúc."""
    cfg = _cfg(tmp_path)
    sch = _sched(cfg)
    try:
        sch.submit([Task(kind="probe", payload={"sleep_ms": 200, "tag": f"c{i}"}) for i in range(8)])
    finally:
        sch.shutdown()
    rows = [r for r in _log(cfg) if r["kind"] == "probe"]
    assert len(rows) == 8 and _max_concurrency(rows) <= 2


def test_four_branches_feed_queue_without_raising_gpu_concurrency(tmp_path, two_gpu):
    """(26) max_branches=4: bốn nhánh cùng đẩy task, nhưng GPU đồng thời vẫn = 2 (không oversubscribe)."""
    cfg = _cfg(tmp_path)
    sch = _sched(cfg)
    errs = []

    def branch(name):
        try:
            scheduler.set_branch(name)
            sch.submit([Task(kind="probe", payload={"sleep_ms": 150, "tag": f"{name}{i}"}) for i in range(3)])
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=branch, args=(f"loop:m{i}",)) for i in range(4)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sch.shutdown()
    assert not errs
    rows = [r for r in _log(cfg) if r["kind"] == "probe"]
    assert len(rows) == 12
    assert _max_concurrency(rows) <= 2  # số worker GPU quyết định, KHÔNG phải số nhánh
    assert len({r["branch"] for r in rows}) == 4  # cả bốn nhánh đều được phục vụ (không starvation)
    assert {r["gpu_physical_id"] for r in rows} == {0, 1}


def test_config_execution_knobs_are_not_in_scientific_hash(tmp_path):
    """max_branches/gpu_devices/gpu_slots_per_device/defer_champion chỉ là thực thi → không đổi config_hash."""
    from p0.config import RunConfig

    base = _cfg(tmp_path).to_dict()
    base.pop("root", None)
    a = RunConfig(**{**base, "eval_seeds": tuple(base["eval_seeds"]), "root": str(tmp_path)})
    b = RunConfig(**{**base, "eval_seeds": tuple(base["eval_seeds"]), "root": str(tmp_path),
                     "max_branches": 4, "gpu_devices": [0, 1], "gpu_slots_per_device": 1, "defer_champion": True})
    assert a.hash() == b.hash()


def test_gpu_stop_inside_a_branch_keeps_user_decision_exit_code(tmp_path, capsys):
    """(22) `gpu_stop` xảy ra TRONG một nhánh của orchestrate vẫn thoát bằng exit 3 (chờ user quyết),
    không bị hạ cấp thành lỗi thường — và thông điệp chỉ in MỘT lần."""
    from p0.checker_log import GPU_STOP_EXIT, gpu_stop
    from p0.orchestrate import build_dag, run_dag

    exp = tmp_path / "experiments"
    branches = build_dag(["lgbm", "xgb"])

    def run_branch(b):
        if b.name == "loop:xgb":
            gpu_stop(exp, "loop", "GPU_RESOURCE_FAILURE", "GPU 1 hết VRAM giữa add-one", model="xgb")
        time.sleep(0.05)

    with pytest.raises(SystemExit) as e:
        run_dag(branches, run_branch, max_active=2)
    assert e.value.code == GPU_STOP_EXIT == 3
    out = capsys.readouterr().out
    assert out.count("DỪNG AN TOÀN") == 1 and "CPU fallback" in out
    rows = [r for r in read_checker(exp) if r["check_id"] == "GPU_RESOURCE_FAILURE"]
    assert rows and rows[-1]["ref"] == "USER_DECISION_REQUIRED"
    assert dict((b.name, b.status) for b in branches)["loop:xgb"] == "error"


def test_gpu_probe_requires_every_configured_gpu_to_be_probed(tmp_path, two_gpu, monkeypatch):
    """(19) nếu không probe đủ mọi worker/GPU thì KHÔNG được kết luận PASS — dừng và hỏi user."""
    cfg = _cfg(tmp_path)
    only_one = [_fake_probe_result(0, 0, "uuid-A", OK_BACKENDS)]  # scheduler giả chỉ trả về 1 worker
    _patch_probe_scheduler(monkeypatch, only_one)
    monkeypatch.setattr("p0.gpu.worker_slots", lambda cfg=None: ([0, 1], 1, 1))  # submit 1 task nhưng cấu hình 2 GPU
    with pytest.raises(SystemExit) as e:
        cli.cmd_gpu_probe(cfg, Namespace(allow_cpu=True, backends=None))
    assert e.value.code == 3
    rows = [r for r in read_checker(cfg.exp_dir) if r["check_id"] == "GPU_PROBE_COVERAGE"]
    assert rows and rows[-1]["ref"] == "USER_DECISION_REQUIRED"
