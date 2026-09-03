"""Vòng expanded-data (quyết định user 2026-09-03) — khoá đúng các yêu cầu §13 của migration:

1. S0_m dựng từ artifact thắng vòng trước (không gõ tay)          2. lọc trùng chính xác + trùng giá trị (semantic)
3. cùng indicator khác lag KHÔNG phải trùng                        4. Candidate_m = C_short \\ overlap(C_short, S0_m)
5. DROP cũ không quay lại pool                                     6. cột S0 khoá không thể bị prune PI bỏ
7. C_short causal                                                  8/9. fold-parallel == tuần tự, thứ tự fold tất định
10. không CPU fallback                                             16. lệnh training không vẽ figure
17. visualize dựng lại figure từ artifact, không train/inference   18. experiments/** không bị gitignore
19. TEST chỉ được chạm ở `final`
"""
import ast
import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p0 import cli, fold_parallel
from p0.config import RunConfig
from p0.features_ext import ALL_EXT_COLUMNS
from p0.features_short import SHORT_BY_NAME, SHORT_CANDIDATES, SHORT_COLUMNS, SHORT_FAMILIES, compute_short
from p0.harness import ColSet, run_config
from p0.loop import confirm, prune_pi
from p0.s0 import collision_audit, prev_dropped, s0_for, save_lock, load_lock

from test_harness_loop import DummyModel

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "experiments" / "15d"


# ----------------------------------------------------------------------------- (1) S0_m từ artifact thắng
@pytest.mark.skipif(not (PREV / "wins").exists(), reason="artifact vòng 15 ngày không có")
def test_s0_reconstructed_from_previous_winners():
    star = ColSet.load(PREV / "b0_star.json")
    for m in ("lgbm", "xgb", "cat", "xgbrf", "lstm", "autots_wr", "autots_mr"):
        w = json.loads((PREV / "wins" / f"{m}.json").read_text(encoding="utf-8"))
        s0 = s0_for(m, PREV)
        assert tuple(s0.b0) == tuple(star.b0) and len(s0.b0) == 72
        assert tuple(s0.ext) == tuple(w["colset"]["ext"]) and s0.locked == s0.ext  # F_old_m KHOÁ toàn bộ
        assert not s0.new_ext
    # AutoTS: mỗi nhánh kế thừa ĐÚNG bộ của nhánh đó (không phải AutoTS-final)
    assert s0_for("autots_wr", PREV).ext != s0_for("autots_mr", PREV).ext
    fin = json.loads((PREV / "wins" / "autots.json").read_text(encoding="utf-8"))["colset"]["ext"]
    assert tuple(s0_for("autots_mr", PREV).ext) != tuple(fin)
    # TimesFM: không bịa covariate kế thừa (TimesFM-final cũ = native)
    assert s0_for("tfm", PREV) == ColSet((), ())


def test_s0_without_previous_run_falls_back_to_b0star_only():
    star = ColSet(("fine:t:candle_return", "coarse:t:rv8"))
    assert s0_for("lgbm", None, star) == ColSet(star.b0)
    with pytest.raises(ValueError):
        s0_for("lgbm", None, None)


# ----------------------------------------------------------------------------- (2)(3)(4)(5) collision + Candidate_m
def test_exact_and_semantic_duplicates_filtered_but_lag_is_not(store):
    from p0 import s0 as s0mod

    # Alias giá trị: cột ext "ret_1" giả = ĐÚNG fine:t:return1 của B0 → phải bị loại; cột khác lag không bị coi là trùng
    alias = SHORT_COLUMNS[0]
    store.ensure_ext((alias,))
    saved = store._ext[alias].copy()
    pos = store._b0_pos["fine:t:return1"]
    idx_all = np.flatnonzero(store.eligible)
    full = np.full(len(store.ts), np.nan, np.float32)
    full[idx_all] = store.matrix(idx_all, store.all_b0())[:, pos]
    store._ext[alias] = full
    try:
        s0 = {"m": ColSet(("fine:t-4m:rsi15_centered", "fine:t:return1"))}
        rep = collision_audit(store, s0, (alias, SHORT_COLUMNS[1], SHORT_COLUMNS[2]), dataset_label="synthetic")
        ident = {r["short"]: r for r in rep["identical"]}
        assert alias in ident and ident[alias]["match"] == "fine:t:return1" and ident[alias]["kind"] == "B0-306"
        assert alias in rep["excluded_from_pool"] and alias not in rep["per_model"]["m"]["candidates"]
        assert SHORT_COLUMNS[1] in rep["per_model"]["m"]["candidates"]
        # rsi15 tại lag 4 (fine:t-4m) và rsi ngắn tại t: KHÔNG trùng (khác timestamp) — không có cặp identical nào khác
        assert all(r["short"] == alias for r in rep["identical"])
    finally:
        store._ext[alias] = saved


def test_candidate_m_is_short_pool_minus_overlap_with_s0(store):
    # S0 chứa sẵn một cột C_short theo TÊN → bị loại khỏi Candidate_m của model đó, vẫn là candidate của model khác
    a, b = SHORT_COLUMNS[3], SHORT_COLUMNS[4]
    s0 = {"m1": ColSet(store.b0_names[:3], (a,), (a,)), "m2": ColSet(store.b0_names[:3])}
    rep = collision_audit(store, s0, (a, b), dataset_label="synthetic")
    assert rep["per_model"]["m1"]["candidates"] == [b] and rep["per_model"]["m1"]["removed_by_overlap"][0]["col"] == a
    assert rep["per_model"]["m2"]["candidates"] == [a, b]


def test_short_pool_excludes_old_candidates_and_dropped_features():
    assert not set(SHORT_COLUMNS) & set(ALL_EXT_COLUMNS)  # 39 candidate cũ (KEEP lẫn DROP) không nằm trong C_short
    if (PREV / "keepdrop_lgbm.csv").exists():
        for m in ("lgbm", "xgbrf", "autots_mr", "cat"):
            assert not set(prev_dropped(PREV, m)) & set(SHORT_COLUMNS)
    # mọi cửa sổ trong lưới ≤ 15 phút; họ H (Keltner) bị bỏ có lý do ghi rõ
    for c in SHORT_CANDIDATES:
        assert len(c.columns) == 1 and c.group in "ABCDEFGHIJKLMNO"
    assert SHORT_FAMILIES["H_kcw"]["use"] == () and all("suy biến" in v for v in SHORT_FAMILIES["H_kcw"]["skip"].values())
    assert len(SHORT_CANDIDATES) == len(set(SHORT_COLUMNS))


def test_save_and_load_lock_roundtrip(tmp_path, store):
    s0 = {"lgbm": ColSet(store.b0_names[:4], ("ret_60",), ("ret_60",))}
    rep = collision_audit(store, s0, SHORT_COLUMNS[:6], dataset_label="synthetic")
    save_lock(tmp_path, s0, rep)
    cs, cands = load_lock(tmp_path, "lgbm")
    assert cs == s0["lgbm"] and [c.name for c in cands] == rep["per_model"]["lgbm"]["candidates"]
    assert (tmp_path / "s0" / "collisions.json").exists() and (tmp_path / "s0" / "short_pool.json").exists()
    with pytest.raises(FileNotFoundError):
        load_lock(tmp_path, "xgb")


# ----------------------------------------------------------------------------- (6) cột khoá không bị prune
def test_locked_ext_cannot_be_pruned_or_removed(store, folds):
    cs = ColSet(store.b0_names[:6], ("ret_60", "bb_pctb_20", SHORT_COLUMNS[0]), ("ret_60", "bb_pctb_20"))
    assert cs.new_ext == (SHORT_COLUMNS[0],)
    with pytest.raises(ValueError):
        cs.without_ext(["ret_60"])
    with pytest.raises(ValueError):
        ColSet(("a",), ("b",), ("c",))  # locked ⊄ ext
    rounds = {f.name: (5, 5, 5) for f in folds}
    pruned, df = prune_pi(store, DummyModel(), cs, folds, rounds, seed=1)
    assert set(df["col"]) == {SHORT_COLUMNS[0]}  # PI chỉ tính cho cột MỚI
    assert set(pruned.locked) == {"ret_60", "bb_pctb_20"} and set(pruned.locked) <= set(pruned.ext)
    d = cs.to_dict()
    assert d["locked"] == ["ret_60", "bb_pctb_20"] and ColSet.from_dict(d) == cs
    assert "locked" not in ColSet(("a",), ("b",)).to_dict()  # tương thích artifact cũ (không có khoá)


# ----------------------------------------------------------------------------- (7) causal
def test_short_candidates_are_causal(store):
    g = store.grid
    t = 4000
    full = compute_short(g).iloc[t]
    cut = compute_short(g.iloc[: t + 1]).iloc[-1]
    for c in SHORT_COLUMNS:
        if np.isnan(full[c]) and np.isnan(cut[c]):
            continue
        assert np.isclose(full[c], cut[c], rtol=1e-5, atol=1e-7), c
    df = compute_short(g)
    assert np.isfinite(df.iloc[3000:].to_numpy()).all()  # sau warmup: hữu hạn (không inf)


# ----------------------------------------------------------------------------- (8)(9)(10) fold-parallel
def _parallel_cfg(tmp_path, hf=None, lf=None) -> RunConfig:
    from p0.synthetic import make_hf, make_lf

    hf = make_hf(n_days=6.0, seed=3)  # 6 ngày: 2 fold VAL (01-03, 01-04) + TEST 01-05 (fixture 4 ngày không đủ cho TEST)
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
        assert par.fold_names == seq.fold_names == [f.name for f in folds]  # thứ tự fold tất định
        assert np.allclose(par.rmse, seq.rmse, rtol=1e-6, atol=1e-6) and np.allclose(par.e0, seq.e0)
        assert (par.best_iters == seq.best_iters).all() and par.rounds == seq.rounds
        # confirmation qua worker (chỉ cần prediction): ŷ giống hệt, latency đo trong worker
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


def test_no_cpu_training_fallback(tmp_path):
    real = RunConfig(dataset_label="btc_full", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"], test_start="2026-01-04", root=str(tmp_path))
    with pytest.raises(SystemExit):
        cli.model_for(real, "lgbm", allow_cpu=True)  # data thật: không được ép CPU
    from p0.models_tfm import TimesFMLoRAModel

    with pytest.raises(RuntimeError):
        TimesFMLoRAModel(device="cpu", allow_cpu=False)
    with pytest.raises(RuntimeError):
        cli.make_model("xgb", {"device": "cpu"}, allow_cpu=False)
    assert fold_parallel.workers_configured(real) == 1


# ----------------------------------------------------------------------------- (16) không vẽ trong training; (19) TEST gating
def _cli_functions():
    tree = ast.parse((ROOT / "src" / "p0" / "cli.py").read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}, tree


def test_training_commands_do_not_plot():
    fns, tree = _cli_functions()
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for n in top_imports:
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


# ----------------------------------------------------------------------------- (17) visualize từ artifact, không train
def test_visualize_regenerates_from_artifacts_without_training(tmp_path, store, folds, monkeypatch):
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
    monkeypatch.setattr(visualize, "cell_metrics", visualize.cell_metrics)  # metric từ ŷ đã lưu là hợp lệ
    made = visualize.regenerate_all(store, folds, final, exp, tmp_path / "figs")
    names = {p.name for p in made}
    assert "fig_path_xgb_vs_champion.png" in names and "fig_HM_xgb_vs_champion.png" in names and "fig_traj_h3_xgb_vs_champion.png" in names
    assert "fig_final_heatmaps.png" in names and "fig_final_paths_all_models.png" in names and "fig_final_traj_h1_all_models.png" in names
    assert all(p.exists() and p.stat().st_size > 0 for p in made)


# ----------------------------------------------------------------------------- (18) experiments/** không bị ignore
def test_experiments_not_gitignored():
    lines = [ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    assert not any(ln.lstrip("/").startswith("experiments") for ln in lines), lines
    for probe in ("experiments/full/runs/x/run.json", "experiments/full/lora/a.pt", "experiments/full/cache/z.npz", "experiments/bootstrap.log"):
        r = subprocess.run(["git", "check-ignore", "-q", probe], cwd=ROOT, capture_output=True)
        if r.returncode not in (0, 1):
            pytest.skip("git không khả dụng")
        assert r.returncode == 1, f"{probe} đang bị ignore"
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "experiments/**/*.npz filter=lfs" in attrs and "experiments/**/*.pt filter=lfs" in attrs


# ----------------------------------------------------------------------------- lock-s0 qua CLI (không có vòng trước → B0*)
def test_cmd_lock_s0_without_prev(tmp_path):
    cfg = _parallel_cfg(tmp_path)
    ColSet(tuple(cli.load_store(cfg)[0].b0_names[:5])).save(cfg.exp_dir / "b0_star.json")
    cfg.model_order = ["lgbm", "tfm"]
    cli.cmd_lock_s0(cfg, Namespace(data_config=None, max_rows=2000))
    cs, cands = load_lock(cfg.exp_dir, "lgbm")
    assert len(cs.b0) == 5 and not cs.ext and len(cands) >= 90
    tcs, tc = load_lock(cfg.exp_dir, "tfm")
    assert tcs == ColSet((), ()) and len(tc) == len(cands)
    rep = json.loads((cfg.exp_dir / "s0" / "collisions.json").read_text(encoding="utf-8"))
    assert rep["audit_dataset_label"] == "synthetic_par" and "near" in rep and "identical" in rep
