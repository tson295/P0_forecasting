"""Vòng expanded-data (quyết định user 2026-09-03, hiệu chỉnh 2026-09-04) — khoá các yêu cầu §20 của pass hiệu chỉnh:

S0/candidate: (1) S0_m từ artifact thắng; (2) TOÀN BỘ S0 khoá (locked_b0 == b0, locked_ext == ext cũ); (3)(4) không bỏ được B0 /
ext khoá; (5)(6) Candidate_m tính riêng từng model = C_short \\ overlap(S0_m); (7) cột B0-306 ngoài S0 KHÔNG chặn toàn cục;
(8) candidate cũ không quay lại; (9) khác lag ≠ trùng; (10) trùng giá trị cùng timestamp → bỏ; (11) tương quan cao KHÔNG bỏ.
C_short: (12) lưới dày ≤ 15; (13) Keltner ngắn; (14) PSAR cửa sổ reset causal; (15) DOW ngoại lệ; (16) log_rv{k}_med2d;
(17) r5_2/r5_3 as-of bar 5' đã đóng; (18) log_c5_ema5_2/3; (19) không EMA1 gap; (20) causal.
Khác: (32) fold-parallel == tuần tự; (33) không vẽ trong training; (34) experiments/** không ignore; (35)(36)(37) checker_log
không tương tác; (38) GPU hard-guard; (39) TEST lần hai bị từ chối; (40) visualize đọc artifact TEST, không inference.
"""
import ast
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p0 import checker_log, cli, fold_parallel
from p0.config import RunConfig
from p0.data import asof_index
from p0.features_ext import ALL_EXT_COLUMNS, _psar_segment
from p0.features_short import SHORT_BY_NAME, SHORT_CANDIDATES, SHORT_COLUMNS, SHORT_FAMILIES, SHORT_GRID, compute_short, psar_window
from p0.harness import ColSet, run_config
from p0.loop import confirm, prune_pi
from p0.s0 import assert_s0_schema, collision_audit, load_lock, prev_dropped, s0_for, save_lock

from test_harness_loop import DummyModel

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "experiments" / "15d"
HAS_PREV = (PREV / "wins").exists()


# ============================================================================= S0 (1)–(4)
@pytest.mark.skipif(not HAS_PREV, reason="artifact vòng 15 ngày không có")
def test_s0_reconstructed_from_previous_winners_and_entirely_locked():
    star = ColSet.load(PREV / "b0_star.json")
    for m in ("lgbm", "xgb", "cat", "xgbrf", "lstm", "autots_wr", "autots_mr"):
        w = json.loads((PREV / "wins" / f"{m}.json").read_text(encoding="utf-8"))
        s0 = s0_for(m, PREV)
        assert tuple(s0.locked_b0) == tuple(s0.b0) == tuple(star.b0) and len(s0.b0) == 72  # mọi B0* khoá
        assert tuple(s0.locked_ext) == tuple(s0.ext) == tuple(w["colset"]["ext"])  # mọi ext thắng cũ khoá
        assert not s0.new_ext and all(s0.is_locked(c) for c in s0.names)
        d = s0.to_dict()
        assert d["locked_b0"] == list(s0.b0) and d["locked_ext"] == list(s0.ext)
        assert_s0_schema(s0, d, m)
    assert s0_for("autots_wr", PREV).ext != s0_for("autots_mr", PREV).ext  # mỗi nhánh AutoTS kế thừa bộ của chính nó
    fin = json.loads((PREV / "wins" / "autots.json").read_text(encoding="utf-8"))["colset"]["ext"]
    assert tuple(s0_for("autots_mr", PREV).ext) != tuple(fin)
    assert s0_for("tfm", PREV) == ColSet((), ())  # TimesFM: không B0*, không covariate kế thừa


def test_colset_schema_explicit_lock_and_backward_compat():
    cs = ColSet(("a", "b"), ("x", "y", "z"), ("x", "y"))
    d = cs.to_dict()
    assert d == {"b0": ["a", "b"], "ext": ["x", "y", "z"], "locked_b0": ["a", "b"], "locked_ext": ["x", "y"]}
    assert ColSet.from_dict(d) == cs and cs.new_ext == ("z",) and cs.locked == ("x", "y")
    assert ColSet.from_dict({"b0": ["a"], "ext": ["x"]}) == ColSet(("a",), ("x",))  # artifact 15 ngày (không có khoá)
    assert ColSet.from_dict({"b0": ["a"], "ext": ["x"], "locked": ["x"]}) == ColSet(("a",), ("x",), ("x",))  # schema 2026-09-03
    with pytest.raises(ValueError):
        ColSet.from_dict({"b0": ["a", "b"], "ext": [], "locked_b0": ["a"], "locked_ext": []})  # B0 khoá không trọn
    with pytest.raises(ValueError):
        assert_s0_schema(ColSet(("a",), ("x",), ()), {"b0": ["a"], "ext": ["x"], "locked_b0": ["a"], "locked_ext": []}, "m")


def test_locked_b0_and_locked_ext_cannot_be_removed(store, folds):
    cs = ColSet(store.b0_names[:6], ("ret_60", "bb_pctb_20", SHORT_COLUMNS[0]), ("ret_60", "bb_pctb_20"))
    with pytest.raises(ValueError):
        cs.without_ext(["ret_60"])  # ext khoá
    with pytest.raises(ValueError):
        cs.without_b0([store.b0_names[0]])  # B0 luôn khoá
    with pytest.raises(ValueError):
        ColSet(("a",), ("b",), ("c",))  # locked_ext ⊄ ext
    rounds = {f.name: (5, 5, 5) for f in folds}
    pruned, df = prune_pi(store, DummyModel(), cs, folds, rounds, seed=1)
    assert set(df["col"]) == {SHORT_COLUMNS[0]}  # PI chỉ xét cột MỚI
    assert pruned.b0 == cs.b0 and set(pruned.locked_ext) == {"ret_60", "bb_pctb_20"} and set(pruned.locked_ext) <= set(pruned.ext)


# ============================================================================= Candidate_m (5)–(11)
def _alias(store, name, values):
    """Ghi đè cache ext để cột C_short `name` mang giá trị tuỳ ý (mô phỏng trùng/affine)."""
    store.ensure_ext((name,))
    saved = store._ext[name].copy()
    store._ext[name] = values.astype(np.float32)
    return saved


def _b0_column_on_grid(store, col):
    idx_all = np.flatnonzero(store.eligible)
    full = np.full(len(store.ts), np.nan, np.float32)
    full[idx_all] = store.matrix(idx_all, store.all_b0())[:, store._b0_pos[col]]
    return full


def test_candidate_m_per_model_exact_and_semantic_overlap_only_against_own_s0(store):
    """(5)(6)(7)(9)(10): overlap CHỈ so với S0 của chính model; cột B0-306 ngoài S0 không chặn; khác lag không trùng."""
    alias, other = SHORT_COLUMNS[0], SHORT_COLUMNS[1]
    saved = _alias(store, alias, _b0_column_on_grid(store, "fine:t:return1"))  # alias == fine:t:return1 (giá trị giống hệt)
    try:
        s0 = {"has_ret1": ColSet(("fine:t:return1", "fine:t-4m:rsi15_centered")),  # S0 chứa return1 tại t
              "no_ret1": ColSet(("fine:t-63m:return1", "fine:t-4m:rsi15_centered")),  # chỉ có return1 ở lag 63 (khác timestamp)
              "by_name": ColSet(("fine:t-63m:return1",), (other,), (other,))}  # S0 chứa `other` theo TÊN
        rep = collision_audit(store, s0, (alias, other, "rsi15_centered"), dataset_label="synthetic")
        pm = rep["per_model"]
        assert pm["has_ret1"]["candidates"] == [other, "rsi15_centered"]  # (10) trùng giá trị cùng timestamp → bỏ
        assert pm["has_ret1"]["removed_by_overlap"][0]["match"] == "fine:t:return1"
        assert pm["no_ret1"]["candidates"] == [alias, other, "rsi15_centered"]  # (7) return1 tại t KHÔNG trong S0 → không chặn
        assert pm["by_name"]["candidates"] == [alias, "rsi15_centered"] and pm["by_name"]["removed_by_overlap"][0]["reason"].startswith("trùng tên")
        # (9) rsi15 tại t vs fine:t-4m:rsi15_centered trong S0: khác timestamp → giữ ở mọi model
        assert all("rsi15_centered" in pm[m]["candidates"] for m in pm)
        assert "identical" not in rep and "excluded_from_pool" not in rep  # không còn lọc toàn cục
    finally:
        store._ext[alias] = saved


def test_high_correlation_is_reported_not_removed(store):
    """(11): cột C_short = 1.5·(B0 return1) + hằng: corr = 1 với cột S0 nhưng KHÔNG trùng giá trị → vẫn là candidate, chỉ báo near."""
    alias = SHORT_COLUMNS[2]
    saved = _alias(store, alias, 1.5 * _b0_column_on_grid(store, "fine:t:return1") + 1e-3)
    try:
        rep = collision_audit(store, {"m": ColSet(("fine:t:return1",))}, (alias,), dataset_label="synthetic")
        pm = rep["per_model"]["m"]
        assert pm["candidates"] == [alias] and not pm["removed_by_overlap"]
        assert any(n["a"] == alias and n["b"] == "fine:t:return1" and abs(n["corr"]) >= 0.995 for n in pm["near_vs_s0"])
    finally:
        store._ext[alias] = saved


def test_old_candidates_never_resurrected():
    assert not set(SHORT_COLUMNS) & set(ALL_EXT_COLUMNS)  # (8) 39 candidate cũ (KEEP lẫn DROP) không nằm trong C_short
    if (PREV / "keepdrop_lgbm.csv").exists():
        for m in ("lgbm", "xgbrf", "autots_mr", "cat"):
            assert not set(prev_dropped(PREV, m)) & set(SHORT_COLUMNS)
    for skip in ("vwap_amt_gap_1", "vwap_amt_gap_15", "ad_vwclv_5", "ad_vwclv_15", "r5_1", "r5_12", "log_c5_ema5_12", "kcw_20"):
        assert skip not in SHORT_COLUMNS


def test_save_and_load_lock_roundtrip_and_schema_guard(tmp_path, store):
    s0 = {"lgbm": ColSet(store.b0_names[:4], ("ret_60",), ("ret_60",))}
    rep = collision_audit(store, s0, SHORT_COLUMNS[:6], dataset_label="synthetic")
    save_lock(tmp_path, s0, rep)
    raw = json.loads((tmp_path / "s0" / "lgbm.json").read_text(encoding="utf-8"))
    assert raw["locked_b0"] == list(store.b0_names[:4]) and raw["locked_ext"] == ["ret_60"]
    cs, cands = load_lock(tmp_path, "lgbm", dataset_label="synthetic")
    assert cs == s0["lgbm"] and [c.name for c in cands] == rep["per_model"]["lgbm"]["candidates"]
    cj = json.loads((tmp_path / "s0" / "candidates_lgbm.json").read_text(encoding="utf-8"))
    assert cj["n_c_short"] == 6 and "near_vs_s0_diagnostic_only" in cj and cj["rule"].startswith("Candidate_m = C_short")
    with pytest.raises(ValueError, match="lock-s0"):
        load_lock(tmp_path, "lgbm", dataset_label="other")
    raw["locked_ext"] = []  # artifact bị sửa → malformed
    (tmp_path / "s0" / "lgbm.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_lock(tmp_path, "lgbm")


# ============================================================================= C_short (12)–(20)
def _windows(fam):
    use = SHORT_FAMILIES[fam]["use"]
    return set(u if isinstance(u, int) else u[0] for u in use)


def test_short_grid_is_dense_and_only_degenerate_windows_skipped():
    for fam, spec in SHORT_FAMILIES.items():
        if fam in ("D_log_ema_ema", "O_macd_hist", "R_r5", "S_log_c5_ema5", "T_dow"):
            continue
        assert set(spec["use"]) | {k for k in spec["skip"] if isinstance(k, int)} == set(SHORT_GRID), fam
        for k, why in spec["skip"].items():
            assert any(w in why for w in ("≡ 0", "chia 0", "log 0", "log(0)", "không xác định", "candidate cũ", "đồng nhất thức")), (fam, k, why)
    for must in ("ret_1", "ret_5", "ret_8", "log_rv5_rv60", "rsi15_centered", "rsi1_centered", "log_atr15_c", "mfi15_centered",
                 "hma_slope15_volnorm", "hma_slope1_volnorm", "dd_2", "ru_2", "ret_skew_3", "bb_pctb_2", "log_range_1", "ad_vwclv_1"):
        assert must in SHORT_COLUMNS, must  # không bỏ vì "xấp xỉ", "nhiễu", "chỉnh lưu", hay có trong B0-306 ngoài B0*
    assert {"log_c_ema1", "dd_1", "ru_1", "bb_pctb_1", "bb_logbw_1", "log_rv1_rv60", "ret_skew_1", "ret_skew_2", "psar_dir_1"}.isdisjoint(SHORT_COLUMNS)
    assert len(SHORT_CANDIDATES) == len(set(SHORT_COLUMNS)) >= 160


def test_short_keltner_and_psar_and_regime_and_5min_families_exist():
    assert {f"kcw_{k}" for k in (2, 3, 4, 5, 8, 10, 15)} <= set(SHORT_COLUMNS) and "kcw_1" not in SHORT_COLUMNS  # (13)
    for fam in ("psar_dir", "psar_logdist", "psar_age_log"):  # (14)
        assert {f"{fam}_{k}" for k in (2, 3, 4, 5, 8, 10, 15)} <= set(SHORT_COLUMNS)
    assert SHORT_FAMILIES["T_dow"]["use"] == () and not [c for c in SHORT_COLUMNS if c.startswith("dow")]  # (15) DOW ngoại lệ
    assert "ngoại lệ" in SHORT_FAMILIES["T_dow"]["skip"]["*"]
    assert {f"log_rv{k}_med2d" for k in (2, 3, 4, 5, 8, 10, 15)} <= set(SHORT_COLUMNS) and "log_rv1_med2d" not in SHORT_COLUMNS  # (16)
    assert {"r5_2", "r5_3"} <= set(SHORT_COLUMNS) and {"r5_1", "r5_12"}.isdisjoint(SHORT_COLUMNS)  # (17)
    assert {"log_c5_ema5_2", "log_c5_ema5_3"} <= set(SHORT_COLUMNS) and {"log_c5_ema5_1", "log_c5_ema5_12"}.isdisjoint(SHORT_COLUMNS)  # (18)(19)
    assert "log_c_ema1" not in SHORT_COLUMNS


def test_psar_window_matches_scalar_segment_and_depends_only_on_last_w_bars(store):
    h, l, c = (store.grid[k].to_numpy(float) for k in ("high", "low", "close"))
    for W in (2, 3, 5, 15):
        d, s, ag = psar_window(h, l, c, W)
        for t in (3000, 4321, 5000):
            dd, ss, aa = (np.full(W, np.nan) for _ in range(3))
            _psar_segment(h[t - W + 1:t + 1], l[t - W + 1:t + 1], c[t - W + 1:t + 1], dd, ss, aa, 0.02, 0.02, 0.2)
            assert dd[-1] == d[t] and np.isclose(ss[-1], s[t]) and aa[-1] == ag[t]
        h2 = h.copy()
        h2[: 5000 - W + 1] *= 1.01  # đổi mọi bar TRƯỚC cửa sổ → trạng thái tại t = 5000 không đổi (reset tại đầu cửa sổ)
        d2, s2, ag2 = psar_window(h2, l, c, W)
        assert d2[5000] == d[5000] and np.isclose(s2[5000], s[5000]) and ag2[5000] == ag[5000]
    d1, s1, a1 = psar_window(h, l, c, 1)
    assert np.isnan(d1).all()  # W = 1: không có trạng thái (bỏ có lý do)


def test_5min_families_use_closed_bar_asof_alignment(store):
    lf = store.raw_lf.sort_values("timestamp").reset_index(drop=True)
    logc5 = np.log(lf["close"].to_numpy(float))
    df = compute_short(store.grid, store.raw_lf, columns=("r5_2", "r5_3", "log_c5_ema5_2"))
    ts = store.grid["timestamp"].to_numpy(np.int64)
    j = asof_index(lf["timestamp"].to_numpy(np.int64), ts)
    for t in (4000, 4003, 5559):
        T = j[t]
        assert lf["timestamp"].iloc[T] <= ts[t] < lf["timestamp"].iloc[T] + 300  # bar 5' đã đóng gần nhất
        assert np.isclose(df["r5_2"].iloc[t], logc5[T] - logc5[T - 2], atol=1e-6) and np.isclose(df["r5_3"].iloc[t], logc5[T] - logc5[T - 3], atol=1e-6)
    assert compute_short(store.grid, None, columns=("r5_2",))["r5_2"].isna().all()  # không có LF → NaN, không bịa


def test_short_rv_med2d_formula(store):
    df = compute_short(store.grid, store.raw_lf, columns=("log_rv5_med2d",))
    r1 = np.log(store.grid["close"].astype(float)).diff()
    rv5 = np.sqrt((r1 ** 2).rolling(5, min_periods=5).mean())
    ref = np.log(rv5 / rv5.rolling(2880, min_periods=2880).median())
    t = 4500
    assert np.isclose(df["log_rv5_med2d"].iloc[t], ref.iloc[t], rtol=1e-5)


def test_short_candidates_are_causal(store):
    g = store.grid
    t = 4000
    full = compute_short(g, store.raw_lf).iloc[t]
    cut = compute_short(g.iloc[: t + 1], store.raw_lf).iloc[-1]
    for c in SHORT_COLUMNS:
        if np.isnan(full[c]) and np.isnan(cut[c]):
            continue
        assert np.isclose(full[c], cut[c], rtol=1e-5, atol=1e-7), c
    df = compute_short(g, store.raw_lf)
    assert not np.isinf(df.to_numpy()).any()


# ============================================================================= fold-parallel (32), GPU (38)
def _parallel_cfg(tmp_path) -> RunConfig:
    from p0.synthetic import make_hf, make_lf

    hf = make_hf(n_days=6.0, seed=3)
    lf = make_lf(hf)
    (tmp_path / "data").mkdir()
    hf.to_csv(tmp_path / "data" / "hf.csv", index=False)
    lf.to_csv(tmp_path / "data" / "lf.csv", index=False)
    cfg = RunConfig(dataset_label="synthetic_par", hf_csv="data/hf.csv", lf_csv="data/lf.csv", val_days=["2026-01-03", "2026-01-04"],
                    test_start="2026-01-05", root=str(tmp_path), require_gpu=False,
                    models={"lgbm": {"n_jobs": 1, "n_estimators": 12, "min_child_samples": 20}})
    cli.cmd_check_data(cfg, Namespace(write_checksums=True))
    return cfg


def test_fold_parallel_matches_sequential_and_keeps_order(tmp_path, monkeypatch):
    cfg = _parallel_cfg(tmp_path)
    store, folds, _, _ = cli.load_store(cfg)
    model = cli.model_for(cfg, "lgbm", allow_cpu=True)
    cs = ColSet(store.b0_names[:12], ("ret_60",))
    rounds = {f.name: (6, 6, 6) for f in folds}
    seq = run_config(store, model, cs, folds, rounds=rounds, seed=3, keep_states=False)
    seq_conf = confirm(store, model, cs, folds, (3, 4), keep_states=True)
    monkeypatch.setenv("P0_FOLD_WORKERS", "2")
    try:
        assert fold_parallel.configure(cfg, model, "lgbm", True) == 2 and fold_parallel.active(model)
        par = run_config(store, model, cs, folds, rounds=rounds, seed=3, keep_states=False)
        assert par.fold_names == seq.fold_names == [f.name for f in folds]
        assert np.allclose(par.rmse, seq.rmse, rtol=1e-6, atol=1e-6) and np.allclose(par.e0, seq.e0)
        assert (par.best_iters == seq.best_iters).all() and par.rounds == seq.rounds
        par_conf = confirm(store, model, cs, folds, (3, 4), keep_states=True, latency_origins=20, measure_latency=True)
        for a, b in zip(seq_conf.runs, par_conf.runs):
            for (ia, ya), (ib, yb) in zip(a.preds(), b.preds()):
                assert np.array_equal(ia, ib) and np.allclose(ya, yb, atol=1e-6)
        assert np.allclose(par_conf.rmse_mean, seq_conf.rmse_mean, atol=1e-6)
        assert par_conf.latency and {r["h"] for r in par_conf.latency} == {1, 2, 3}
    finally:
        fold_parallel.shutdown()
        monkeypatch.delenv("P0_FOLD_WORKERS", raising=False)
        fold_parallel.configure(cfg, model, "lgbm", True)
    assert not fold_parallel.active(model)


def _real_cfg(tmp_path):
    (tmp_path / ".claude").mkdir(exist_ok=True)
    (tmp_path / ".claude" / "MEMORY.md").write_text("TRAINING: UNLOCKED\n", encoding="utf-8")
    return RunConfig(dataset_label="btc_full", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"], test_start="2026-01-04", root=str(tmp_path))


def test_gpu_only_is_a_hard_non_interactive_guard(tmp_path, monkeypatch):
    real = _real_cfg(tmp_path)
    with pytest.raises(SystemExit):
        cli.model_for(real, "lgbm", allow_cpu=True)  # data thật: không được ép CPU
    with pytest.raises(RuntimeError):
        cli.make_model("xgb", {"device": "cpu"}, allow_cpu=False)
    from p0.models_tfm import TimesFMLoRAModel

    with pytest.raises(RuntimeError):
        TimesFMLoRAModel(device="cpu", allow_cpu=False)
    # preflight fail → gate dừng NGAY bằng SystemExit + ERROR trong checker_log (không hỏi user)
    monkeypatch.setattr(cli, "gpu_preflight", lambda m, c: (_ for _ in ()).throw(SystemExit("GPU preflight xgb: không có CUDA")))
    with pytest.raises(SystemExit):
        cli.gate(real, Namespace(smoke=False, allow_cpu=False), ["xgb"])
    rows = checker_log.read(real.exp_dir)
    assert rows and rows[-1]["severity"] == "ERROR" and rows[-1]["check_id"] == "GPU_UNAVAILABLE" and rows[-1]["model"] == "xgb"
    # XGBoost: wheel không CUDA → preflight từ chối (không chỉ "yêu cầu" device=cuda)
    import xgboost as xgb

    monkeypatch.setattr(xgb, "build_info", lambda: {"USE_CUDA": False})
    with pytest.raises(SystemExit, match="CUDA"):
        cli.gpu_preflight("xgb", real)


# ============================================================================= checker_log (35)(36)(37)
def test_checker_log_persists_findings_without_prompt(tmp_path):
    exp = tmp_path / "exp"
    row = checker_log.record(exp, "pre-run", "WARN", "CORR_HIGH", "ret_2 ~ ret_3 corr 0.99", model="lgbm", file="s0.py", ref="L10")
    assert set(row) == {"timestamp", "stage", "model", "severity", "check_id", "message", "file", "ref"}
    checker_log.record(exp, "pre-run", "INFO", "RUNTIME", "ok")  # WARN/INFO không dừng gì
    assert [r["severity"] for r in checker_log.read(exp)] == ["WARN", "INFO"]
    with pytest.raises(ValueError):
        checker_log.record(exp, "x", "FATAL", "y", "z")
    with pytest.raises(SystemExit):  # bất biến cứng → thoát ngay, ghi ERROR
        checker_log.hard_fail(exp, "pre-run", "CHECKSUM_MISMATCH", "sha khác")
    assert checker_log.blocking_errors(exp)[0]["check_id"] == "CHECKSUM_MISMATCH"
    checker_log.record(exp, "pre-run", "PASS", "CHECKSUM_MISMATCH", "đã sửa")
    assert checker_log.blocking_errors(exp) == []
    r = subprocess.run(["python", str(ROOT / "scripts" / "checker_record.py"), "--exp", str(exp), "--stage", "code-review", "--severity", "WARN",
                        "--check-id", "X", "--message", "m"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and checker_log.read(exp)[-1]["check_id"] == "X"


def test_training_lock_and_cpu_flag_hard_fail_with_error_record(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "MEMORY.md").write_text("TRAINING: LOCKED\n", encoding="utf-8")
    real = RunConfig(dataset_label="btc_full", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"], test_start="2026-01-04", root=str(tmp_path))
    with pytest.raises(SystemExit):
        cli.gate(real, Namespace(smoke=False, allow_cpu=False), ["lgbm"])
    with pytest.raises(SystemExit):
        cli.gate(real, Namespace(smoke=True, allow_cpu=False), ["lgbm"])
    ids = [r["check_id"] for r in checker_log.read(real.exp_dir)]
    assert ids == ["TRAINING_LOCKED", "CPU_ON_REAL_DATA"] and all(r["severity"] == "ERROR" for r in checker_log.read(real.exp_dir))


# ============================================================================= TEST một lần (39), visualize (40), no-plot (33), TEST gating
def test_second_final_is_rejected_without_prompt(tmp_path, monkeypatch):
    real = _real_cfg(tmp_path)
    exp = real.exp_dir
    (exp / "final").mkdir(parents=True)
    (exp / "final" / "TEST_SENTINEL.json").write_text(json.dumps({"status": "completed", "started_at": "t0", "config_hash": real.hash()}), encoding="utf-8")
    monkeypatch.setattr(cli, "gate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gate không được gọi trước sentinel")))
    with pytest.raises(SystemExit, match="TEST_ALREADY_RUN"):
        cli.cmd_final(real, Namespace(smoke=False, allow_cpu=False, latency_origins=None, force_test_rerun=False))
    assert checker_log.read(exp)[-1]["check_id"] == "TEST_ALREADY_RUN"

    class Stop(Exception):
        pass

    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_store", lambda c, **k: (_ for _ in ()).throw(Stop()))
    with pytest.raises(Stop):  # --force-test-rerun (recovery) vượt sentinel, ghi WARN
        cli.cmd_final(real, Namespace(smoke=False, allow_cpu=False, latency_origins=None, force_test_rerun=True))
    assert checker_log.read(exp)[-1]["check_id"] == "TEST_RERUN_FORCED"


def _cli_functions():
    tree = ast.parse((ROOT / "src" / "p0" / "cli.py").read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}, tree


def test_training_commands_do_not_plot():
    fns, tree = _cli_functions()
    for n in [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]:
        names = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
        assert not any(x and ("plots" in x or "matplotlib" in x or "visualize" in x) for x in names), ast.dump(n)
    for name, fn in fns.items():
        if not name.startswith("cmd_") or name == "cmd_visualize":
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert node.value.id not in ("plots", "plt", "visualize"), (name, node.attr)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                assert not any(m and ("plots" in m or "matplotlib" in m or "visualize" in m) for m in mods), (name, mods)


def test_test_partition_only_touched_in_final():
    fns, _ = _cli_functions()
    allowed = {"make_partitions", "load_store", "cmd_check_data", "cmd_final", "cmd_visualize"}
    for name, fn in fns.items():
        if name in allowed:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id == "final":
                raise AssertionError(f"{name} chạm fold TEST (`final`) — chỉ cmd_final/cmd_visualize được phép")


def test_visualize_reads_saved_test_artifacts_without_inference(tmp_path, store, folds, monkeypatch):
    from p0 import visualize
    from p0.split import make_final

    exp = tmp_path / "exp"
    (exp / "wins").mkdir(parents=True)
    a = confirm(store, DummyModel(), ColSet(store.b0_names[:5]), folds, (1, 2), keep_states=True)
    b = confirm(store, DummyModel(), ColSet(store.b0_names[5:10]), folds, (1, 2), keep_states=True)
    for name, conf in (("lgbm", a), ("xgb", b)):
        cli._save_win(exp, name, conf, 0.02, "prune", folds)
    cli.log_champion(exp, {"exp_id": "x", "model": "xgb", "champion_before": "lgbm", "decision": "giữ", "MedianGain_vs_champion": -0.01,
                           "eps_champion": 0.02, "champion_after": "lgbm"})
    final = make_final(store.first_origin_ts, "2026-01-04", store.last_ts + 60, 60)
    idx = final.val.origins(store.ts, store.eligible)
    (exp / "final").mkdir()
    for key in ("lgbm", "xgb"):
        np.savez_compressed(exp / "final" / f"{key}.npz", idx_0=idx, yhat_0=np.zeros((len(idx), 3), np.float32))
    (exp / "final" / "index.json").write_text(json.dumps({"keys": ["lgbm", "xgb"], "models": {}, "champion": "lgbm"}), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("visualize không được train/inference")

    monkeypatch.setattr(cli, "model_for", boom)
    monkeypatch.setattr(cli, "run_config", boom)
    made = visualize.regenerate_all(store, folds, final, exp, tmp_path / "figs")
    names = {p.name for p in made}
    assert {"fig_path_xgb_vs_champion.png", "fig_HM_xgb_vs_champion.png", "fig_final_heatmaps.png", "fig_final_paths_all_models.png",
            "fig_final_traj_h1_all_models.png"} <= names
    assert all(p.exists() and p.stat().st_size > 0 for p in made)


# ============================================================================= git (34), lock-s0 CLI
def test_experiments_not_gitignored():
    lines = [ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    assert not any(ln.lstrip("/").startswith("experiments") for ln in lines), lines
    for probe in ("experiments/full/runs/x/run.json", "experiments/full/lora/a.pt", "experiments/full/cache/z.npz", "experiments/bootstrap.log",
                  "experiments/full/checker_log.jsonl", "experiments/full/final/TEST_SENTINEL.json"):
        r = subprocess.run(["git", "check-ignore", "-q", probe], cwd=ROOT, capture_output=True)
        if r.returncode not in (0, 1):
            pytest.skip("git không khả dụng")
        assert r.returncode == 1, f"{probe} đang bị ignore"
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "experiments/**/*.npz filter=lfs" in attrs and "experiments/**/*.pt filter=lfs" in attrs


def test_cmd_lock_s0_without_prev_writes_per_model_candidates(tmp_path):
    cfg = _parallel_cfg(tmp_path)
    ColSet(tuple(cli.load_store(cfg)[0].b0_names[:5])).save(cfg.exp_dir / "b0_star.json")
    cfg.model_order = ["lgbm", "tfm"]
    cli.cmd_lock_s0(cfg, Namespace(data_config=None, max_rows=2000))
    cs, cands = load_lock(cfg.exp_dir, "lgbm", dataset_label=cfg.dataset_label)
    assert len(cs.b0) == 5 and not cs.ext and len(cands) == len(SHORT_COLUMNS)
    tcs, tc = load_lock(cfg.exp_dir, "tfm", dataset_label=cfg.dataset_label)
    assert tcs == ColSet((), ()) and len(tc) == len(cands)
    rep = json.loads((cfg.exp_dir / "s0" / "collisions.json").read_text(encoding="utf-8"))
    assert rep["audit_dataset_label"] == "synthetic_par" and set(rep["per_model"]) == {"lgbm", "tfm"}
    for m in ("lgbm", "tfm"):
        assert (cfg.exp_dir / "s0" / f"candidates_{m}.json").exists()
    ids = [r["check_id"] for r in checker_log.read(cfg.exp_dir)]
    assert "S0_LOCK" in ids and "CANDIDATE_M" in ids
