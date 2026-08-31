"""Flow TimesFM + AutoTS sau khi chốt methodology (2026-08-31):

- TFM-native chỉ dùng r1; chiến lược covariate resolve MỘT lần rồi freeze, không đổi giữa fold/candidate;
  strategy B0*/subset thì candidate ext cộng LÊN baseline đó; add-one cập nhật S tuần tự.
- WR/MR chỉ là probe (không gọi framework `AutoTS(...)`, không so champion); `autots-union` tạo F_WR_best/F_MR_best;
  framework AutoTS chỉ chạy sau khi feature set đã freeze.
"""
import json
from argparse import Namespace

import numpy as np
import pytest

from p0 import cli
from p0.config import RunConfig
from p0.harness import ColSet, run_config
from p0.loop import add_one_loop
from p0.models import SeriesBatch
from p0.models_tfm import COVARIATE_STRATEGIES, TimesFMModel

from test_tfm_autots import StubTFM  # noqa: F401  (stub dùng lại; tests/ nằm trên sys.path qua conftest)


def _cfg(tmp_path, tfm_params=None, label="synthetic_flow"):
    return RunConfig(dataset_label=label, hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"], test_start="2026-01-04",
                     root=str(tmp_path), models={"tfm": tfm_params} if tfm_params else {})


# ----------------------------------------------------------------------------- G1: native chỉ dùng r1
def test_tfm_native_uses_only_r1(store, folds):
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, model=StubTFM())
    run = run_config(store, m, ColSet((), ()), folds[:1], rounds=None, seed=1)
    st = run.states[0]
    assert isinstance(st.X_val, SeriesBatch)
    assert st.X_val.cov is None and st.X_val.cov_names == ()  # không covariate nào đi vào input
    # đổi cột ext bất kỳ cũng không đổi prediction native
    m2 = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, model=StubTFM())
    run2 = run_config(store, m2, ColSet(store.b0_names[:5], ("ret_60", "bb_pctb_20")), folds[:1], rounds=None, seed=1)
    assert np.allclose(run.states[0].yhat, run_config(store, TimesFMModel(device="cpu", allow_cpu=True, context=512,
                                                                         batch_size=64, model=StubTFM()),
                                                      ColSet(store.b0_names[:9], ()), folds[:1], rounds=None, seed=1).states[0].yhat)
    assert not np.allclose(run.states[0].yhat, run2.states[0].yhat)  # ext_only: có candidate thì prediction phải đổi


# ----------------------------------------------------------------------------- G2: strategy resolve 1 lần rồi freeze
def test_tfm_strategy_frozen_once(tmp_path, store):
    b0star = ColSet(store.b0_names[:20])
    cfg = _cfg(tmp_path, {"covariate_strategy": "ext_only"})
    strategy, base = cli.resolve_tfm_strategy(cfg, store, b0star)
    assert strategy == "ext_only" and base.b0 == () and base.ext == ()
    frozen = json.loads(cli.tfm_strategy_path(cfg).read_text(encoding="utf-8"))
    assert frozen["strategy"] == "ext_only" and frozen["n_covariate_base"] == 0
    # gọi lại cùng config: OK, không đổi artifact
    assert cli.resolve_tfm_strategy(cfg, store, b0star)[0] == "ext_only"
    # đổi strategy giữa chừng: DỪNG (cấm đổi hướng giữa pipeline)
    with pytest.raises(SystemExit):
        cli.resolve_tfm_strategy(_cfg(tmp_path, {"covariate_strategy": "b0star_full"}), store, b0star)
    # thiếu khai báo hoặc tên sai: DỪNG (không có mặc định ngầm)
    for bad in (None, {"covariate_strategy": "whatever"}):
        with pytest.raises(SystemExit):
            cli.resolve_tfm_strategy(_cfg(tmp_path / "b", bad), store, b0star)


def test_tfm_strategy_b0star_and_subset(tmp_path, store):
    b0star = ColSet(store.b0_names[:20])
    cfg = _cfg(tmp_path / "full", {"covariate_strategy": "b0star_full"})
    strategy, base = cli.resolve_tfm_strategy(cfg, store, b0star)
    assert strategy == "b0star_full" and base.b0 == b0star.b0 and base.ext == ()
    # subset: deterministic theo PI của §1.4, giữ thứ tự cột gốc
    cfg2 = _cfg(tmp_path / "sub", {"covariate_strategy": "b0star_subset", "subset_k": 5})
    cfg2.exp_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pi = rng.normal(size=(20, 3))
    import pandas as pd

    pd.DataFrame({"col": list(b0star.b0), **{f"PI_h{h}": pi[:, h - 1] for h in (1, 2, 3)}}).to_csv(
        cfg2.exp_dir / "b0_filter.csv", index=False)
    _, sub = cli.resolve_tfm_strategy(cfg2, store, b0star)
    want = sorted(sorted(range(20), key=lambda i: -pi[i].mean())[:5])
    assert sub.b0 == tuple(b0star.b0[i] for i in want) and len(sub.b0) == 5
    assert cli.resolve_tfm_strategy(cfg2, store, b0star)[1].b0 == sub.b0  # lặp lại cho cùng kết quả


# ----------------------------------------------------------------------------- G3: candidate cộng lên baseline đã chọn
def test_candidate_adds_on_top_of_strategy_baseline(store, folds):
    """b0star_full/subset: covariate = baseline B0* + candidate ext (đúng thứ tự); ext_only: chỉ candidate."""
    base = ColSet(store.b0_names[:6], ())
    cand = base.with_ext(("ret_60",))
    m_all = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64,
                         covariate_strategy="b0star_full", model=StubTFM())
    st = run_config(store, m_all, cand, folds[:1], rounds=None, seed=1).states[0]
    assert st.X_val.cov_names == tuple(cand.names) and st.X_val.cov.shape[1] == 7
    m_ext = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64,
                         covariate_strategy="ext_only", model=StubTFM())
    st2 = run_config(store, m_ext, cand, folds[:1], rounds=None, seed=1).states[0]
    assert st2.X_val.cov_names == ("ret_60",) and st2.X_val.cov.shape[1] == 1  # KHÔNG kéo B0* vào


# ----------------------------------------------------------------------------- G4: add-one cập nhật S tuần tự
def test_sequential_keep_updates_S(store, folds):
    from p0.features_ext import CANDIDATE_BY_NAME
    from test_harness_loop import DummyModel

    base = ColSet(store.b0_names[:6])
    base_run = run_config(store, DummyModel(), base, folds, rounds=None, seed=1)
    cands = [CANDIDATE_BY_NAME[n] for n in ("ret_60", "bb_pctb_20", "rsi240_centered")]
    seen = []
    lr = add_one_loop(store, DummyModel(), base, base_run.rmse, cands, folds, None, 0.05, 1, base_run.e0,
                      None, lambda row, run: seen.append(tuple(run.colset.ext)))
    # candidate thứ i luôn được thử TRÊN S hiện tại: ext của run i = (các candidate đã KEEP) + candidate i
    kept = []
    for i, c in enumerate(cands):
        assert seen[i] == tuple(kept) + c.columns, (i, seen[i], kept)
        if lr.table["decision"].iloc[i] == "KEEP":
            kept += list(c.columns)
    assert tuple(lr.final.ext) == tuple(kept)


# ----------------------------------------------------------------------------- G5: probe không gọi framework AutoTS
def test_probe_models_never_call_autots_framework():
    """Probe chỉ dùng 2 class cố định; KHÔNG có tham chiếu tới framework `AutoTS` trong CODE (bỏ qua comment/docstring)."""
    import ast

    src = (cli.Path(cli.__file__).parent / "models_autots.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "AutoTS" not in imported and "AutoTS" not in names
    assert {"WindowRegression", "MultivariateRegression"} <= imported
    assert cli.PROBE_MODELS == ("autots_wr", "autots_mr")


# ----------------------------------------------------------------------------- G6: union tạo F_WR_best / F_MR_best
def _probe_win(exp, model, ext, rmse):
    (exp / "wins").mkdir(parents=True, exist_ok=True)
    (exp / "wins" / f"{model}.json").write_text(json.dumps(
        {"model": model, "colset": {"b0": ["fine:t:log_ret_1"], "ext": list(ext)}, "rmse_mean": rmse,
         "e0": [[100.0, 140.0, 170.0]], "eps": 0.02, "eval_seeds": [1, 2, 3], "which": "prune",
         "median_gain_vs_e0": 0.1}), encoding="utf-8")


def test_union_builds_best_sets_without_champion(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    rmse = [[50.0, 70.0, 90.0]]
    _probe_win(exp, "autots_wr", ["ret_60"], rmse)
    _probe_win(exp, "autots_mr", ["ret_60"], rmse)  # trùng nhau → không cần run thêm
    monkeypatch.setattr(cli, "load_store", lambda c, **k: (None, [], None, None))
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    cli.cmd_autots_union(cfg, Namespace(smoke=True, allow_cpu=True))
    feats = json.loads((exp / "autots_features.json").read_text(encoding="utf-8"))
    assert feats["F_WR"] == ["ret_60"] and feats["F_MR"] == ["ret_60"]
    assert feats["F_WR_best"] == ["ret_60"] and feats["F_MR_best"] == ["ret_60"] and feats["identical"] is True
    for m in cli.PROBE_MODELS:
        p = json.loads((exp / "autots_best" / f"{m}.json").read_text(encoding="utf-8"))
        assert p["role"] == "probe_best_feature_set" and p["chosen"] == "riêng"
    assert not (exp / "champion.json").exists()  # union KHÔNG đụng champion
    assert not (exp / "champion_log.csv").exists()


def test_strategy_names_are_the_three_audited_options():
    assert COVARIATE_STRATEGIES == ("b0star_full", "b0star_subset", "ext_only")
