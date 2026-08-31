"""Flow feature-selection của TimesFM + AutoTS (rev 2026-08-31: hai nhánh TimesFM, bỏ union AutoTS).

Khoá đúng những gì §8 yêu cầu:
- vòng add-one chung: candidate i chạy trên S hiện tại + f_i, KEEP thì candidate sau thấy feature vừa KEEP,
  và model được CHẠY LẠI mỗi khi feature set đổi (không tái dùng prediction cũ);
- confirmation: model có ES dùng fixed rounds ở add-one/prune nhưng `rounds=None` (ES bật lại) ở confirmation;
- TimesFM: nhánh B0* thật sự giữ cột B0*, nhánh ext thật sự bắt đầu từ ∅ (native), mỗi nhánh prune+confirm riêng,
  TimesFM-final chọn giữa hai nhánh bằng metric project;
- AutoTS: WR/MR đều là probe từ B0*, không còn stage union.
"""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from p0 import cli
from p0.config import RunConfig
from p0.harness import ColSet, run_config
from p0.loop import add_one_loop, confirm, prune_pi
from p0.models import make_model
from p0.models_tfm import COVARIATE_SCOPES, TimesFMModel

from test_tfm_autots import StubTFM


def _cfg(tmp_path, **kw):
    return RunConfig(dataset_label="synthetic_flow", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"],
                     test_start="2026-01-04", root=str(tmp_path), **kw)


# ============================================================================= vòng add-one chung
class RecordingModel:
    """Model tuyến tính tất định, ghi lại (số cột, rounds, seed) của MỖI lần fit → chứng minh model được chạy lại."""

    name = "rec"
    input_kind = "tabular"
    supports_rounds = True
    seed_dependent = True
    train_device = predict_device = "CPU"

    def __init__(self):
        self.calls = []

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds, seed):
        from p0.models import FitResult

        self.calls.append({"n_col": np.asarray(X_fit).shape[1], "rounds": rounds, "seed": seed})
        Xf = np.nan_to_num(np.asarray(X_fit, float))
        W = np.linalg.lstsq(np.c_[Xf, np.ones(len(Xf))], z_fit, rcond=None)[0]
        Xp = np.c_[np.nan_to_num(np.asarray(X_pred, float)), np.ones(len(X_pred))]
        preds = [lambda X, c=c, W=W: (np.c_[np.nan_to_num(np.asarray(X, float)), np.ones(len(X))] @ W[:, c]).astype(np.float32)
                 for c in range(3)]
        return FitResult((Xp @ W).astype(np.float32), tuple(rounds) if rounds is not None else (5, 6, 7), preds)


def test_add_one_reruns_model_on_updated_S(store, folds):
    """Mỗi candidate = một lần fit MỚI trên S hiện tại + f; KEEP thì candidate sau thấy feature vừa KEEP."""
    from p0.features_ext import CANDIDATE_BY_NAME

    base = ColSet(store.b0_names[:6])
    m = RecordingModel()
    rounds = {f.name: (5, 5, 5) for f in folds}
    base_run = run_config(store, m, base, folds, rounds=rounds, seed=1)
    n0 = len(m.calls)
    cands = [CANDIDATE_BY_NAME[n] for n in ("ret_60", "bb_pctb_20", "rsi240_centered")]
    seen = []
    lr = add_one_loop(store, m, base, base_run.rmse, cands, folds, rounds, 0.05, 1, base_run.e0,
                      None, lambda row, run: seen.append(tuple(run.colset.ext)))
    kept = []
    for i, c in enumerate(cands):
        assert seen[i] == tuple(kept) + c.columns  # thử trên S HIỆN TẠI + f_i
        if lr.table["decision"].iloc[i] == "KEEP":
            kept += list(c.columns)
    assert tuple(lr.final.ext) == tuple(kept)
    # model thực sự chạy lại: mỗi candidate = len(folds) lần fit; số cột đầu vào = |S hiện tại| + |f_i|
    calls = m.calls[n0:]
    assert len(calls) == len(cands) * len(folds)
    for i, c in enumerate(cands):
        assert calls[i * len(folds)]["n_col"] == len(base.names) + len(seen[i])
    assert all(c["rounds"] == (5, 5, 5) for c in calls)  # add-one dùng FIXED rounds (ES off)


def test_confirmation_turns_es_back_on(store, folds):
    """add-one/prune: rounds cố định (ES OFF). confirmation: rounds=None → ES BẬT LẠI, refit từ đầu, 3 eval seed."""
    cs = ColSet(store.b0_names[:6], ("ret_60",))
    rounds = {f.name: (4, 4, 4) for f in folds}
    m = RecordingModel()
    prune_pi(store, m, cs, folds, rounds, seed=1, repeats=1)
    assert m.calls and all(c["rounds"] == (4, 4, 4) for c in m.calls)  # prune PI vẫn dùng số vòng cố định
    m2 = RecordingModel()
    conf = confirm(store, m2, cs, folds, (11, 12, 13))
    assert all(c["rounds"] is None for c in m2.calls)  # confirmation: ES bật lại
    assert sorted({c["seed"] for c in m2.calls}) == [11, 12, 13]
    assert len(conf.runs) == 3 and conf.rmse_mean.shape == (len(folds), 3)


# ============================================================================= TimesFM: đúng HAI nhánh
def test_two_branches_only():
    assert COVARIATE_SCOPES == {"b0star": "all", "ext": "ext"}
    assert cli.TFM_BRANCH_BASE == {"tfm_ext": "empty", "tfm_b0": "b0star"}
    assert set(cli.PROBE_MODELS) == {"autots_wr", "autots_mr", "tfm_b0", "tfm_ext"}
    b0 = make_model("tfm_b0", {"device": "cpu"}, allow_cpu=True)
    ext = make_model("tfm_ext", {"device": "cpu"}, allow_cpu=True)
    assert (b0.name, b0.covariate_scope, b0.series_covariates) == ("tfm_b0", "b0star", "all")
    assert (ext.name, ext.covariate_scope, ext.series_covariates) == ("tfm_ext", "ext", "ext")
    assert not hasattr(b0, "covariate_strategy")  # 3-way strategy + b0star_subset đã bỏ


def test_branch_b0_keeps_b0_columns_and_adds_candidate(store, folds):
    """Nhánh A: covariate = B0* + candidate; B0* KHÔNG bị âm thầm loại bỏ, và thêm candidate thì phải chạy lại."""
    base = ColSet(store.b0_names[:6], ())
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope="b0star", model=StubTFM())
    st0 = run_config(store, m, base, folds[:1], rounds=None, seed=1).states[0]
    assert st0.X_val.cov_names == tuple(base.b0) and st0.X_val.cov.shape[1] == 6
    st1 = run_config(store, m, base.with_ext(("ret_60",)), folds[:1], rounds=None, seed=1).states[0]
    assert st1.X_val.cov_names == tuple(base.b0) + ("ret_60",) and st1.X_val.cov.shape[1] == 7
    assert not np.allclose(st0.yhat, st1.yhat)  # forecast + xreg được fit lại với ma trận covariate mới


def test_branch_ext_starts_from_native(store, folds):
    """Nhánh B: S = ∅ → baseline là TimesFM native (chỉ r1); candidate cộng dần; KHÔNG kéo cột B0* vào."""
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope="ext", model=StubTFM())
    st_native = run_config(store, m, ColSet((), ()), folds[:1], rounds=None, seed=1).states[0]
    assert st_native.X_val.cov is None and st_native.X_val.cov_names == ()
    st_c = run_config(store, m, ColSet((), ("ret_60",)), folds[:1], rounds=None, seed=1).states[0]
    assert st_c.X_val.cov_names == ("ret_60",) and st_c.X_val.cov.shape[1] == 1
    st_mix = run_config(store, m, ColSet(store.b0_names[:5], ("ret_60",)), folds[:1], rounds=None, seed=1).states[0]
    assert st_mix.X_val.cov_names == ("ret_60",)


def test_prune_positions_per_branch(store, folds):
    """prune PI trỏ đúng cột ext trong ma trận covariate của TỪNG nhánh (b0star: ext nằm sau b0; ext: từ 0)."""
    for scope, cs in (("b0star", ColSet(store.b0_names[:4], ("ret_60", "bb_pctb_20"))),
                      ("ext", ColSet((), ("ret_60", "bb_pctb_20")))):
        m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope=scope, model=StubTFM())
        pruned, df = prune_pi(store, m, cs, folds[:1], rounds=None, seed=1, repeats=1)
        assert list(df["col"]) == list(cs.ext) and pruned.b0 == cs.b0 and set(pruned.ext) <= set(cs.ext)


def _win(exp, model, b0, ext, rmse, e0=((100.0, 140.0, 170.0),)):
    (exp / "wins").mkdir(parents=True, exist_ok=True)
    (exp / "wins" / f"{model}.json").write_text(json.dumps(
        {"model": model, "colset": {"b0": list(b0), "ext": list(ext)}, "rmse_mean": [list(r) for r in rmse],
         "e0": [list(r) for r in e0], "eps": 0.02, "eval_seeds": [1, 2, 3], "which": "prune",
         "median_gain_vs_e0": 0.1}), encoding="utf-8")


def test_tfm_final_picks_better_branch_by_project_metric(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    _win(exp, "tfm_b0", ["fine:t:log_ret_1"], ["ret_60"], [(90.0, 130.0, 160.0)])   # kém hơn
    _win(exp, "tfm_ext", [], ["ret_60", "bb_pctb_20"], [(80.0, 120.0, 150.0)])      # tốt hơn
    (exp / "champion.json").write_text(json.dumps(
        {"model": "lgbm", "colset": {"b0": [], "ext": []}, "rmse_mean": [[95.0, 135.0, 165.0]], "eps": 0.02,
         "e0": [[100.0, 140.0, 170.0]]}), encoding="utf-8")
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    fin = json.loads((exp / "wins" / "tfm.json").read_text(encoding="utf-8"))
    assert fin["model"] == "tfm" and fin["role"] == "TimesFM-final"
    assert fin["branch"] == "tfm_ext" and fin["covariate_scope"] == "ext"
    df = pd.read_csv(exp / "tfm_final.csv")
    assert set(df["branch"]) == {"tfm_b0", "tfm_ext"} and len(df) == 2
    assert df.loc[df["MedianGain_vs_E0"].idxmax(), "branch"] == "tfm_ext"
    ch = pd.read_csv(exp / "champion_log.csv")
    assert (ch["model"] == "tfm").any()  # TimesFM-final mới là thứ đi so champion


def test_final_step_cannot_become_initial_champion(tmp_path, monkeypatch):
    """§3: champion ban đầu phải là LightGBM — tfm-final/autots-search không được tự thành champion đầu tiên."""
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    _win(exp, "tfm_b0", [], ["ret_60"], [(90.0, 130.0, 160.0)])
    _win(exp, "tfm_ext", [], ["ret_60"], [(80.0, 120.0, 150.0)])
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "loop --model lgbm" in str(e.value)


def test_tfm_final_requires_both_branches(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _win(cfg.exp_dir, "tfm_b0", [], ["ret_60"], [(90.0, 130.0, 160.0)])
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "tfm_ext" in str(e.value)


# ============================================================================= AutoTS: probe, không còn union
def test_no_union_stage_left():
    assert not hasattr(cli, "cmd_autots_union")
    src = cli.Path(cli.__file__).read_text(encoding="utf-8")
    assert "autots-union" not in src and "F_union" not in src and "autots_best" not in src
    for m in ("autots_wr", "autots_mr"):
        assert cli.FINAL_STEP[m] == "autots-search"
        assert m in cli.PROBE_MODELS and cli.TFM_BRANCH_BASE.get(m) is None  # probe start từ B0* (mặc định)


def test_probe_models_never_call_autots_framework():
    """Probe chỉ dùng 2 class cố định; framework `AutoTS` nằm ở module riêng `autots_search.py`."""
    import ast

    src = (cli.Path(cli.__file__).parent / "models_autots.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "AutoTS" not in imported and "AutoTS" not in names
    assert {"WindowRegression", "MultivariateRegression"} <= imported
