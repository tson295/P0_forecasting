"""§2.2 #6 (iii) — bake-off template GPU: framework AutoTS chỉ chạy SAU khi feature set đã freeze, search chỉ thấy
training-side (FIT+ES), hai frozen set search độc lập, và AutoTS-final chọn bằng metric của project (không phải điểm AutoTS).

Package `autots` chưa cài → stub cả `search_best_template` (framework) lẫn class model (ModelMonster).
"""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from p0 import cli
from p0.config import RunConfig
from p0.harness import ColSet

from test_tfm_autots import StubAutoTS


class StubFrozen(StubAutoTS):
    """Thay `ModelMonster(name, parameters=..., ...)`: prediction phụ thuộc feature set để phân biệt hai frozen set."""

    def __init__(self, name=None, parameters=None, **kw):
        super().__init__(**kw)
        self.model_name, self.parameters = name, parameters or {}

    def predict(self, forecast_length=None, future_regressor=None, just_point_forecast=False, **kw):
        self.predict_calls.append((future_regressor.copy(), just_point_forecast, self.regressor_train))
        v = 1e-4 * float(future_regressor.to_numpy()[-1].sum()) / max(1, future_regressor.shape[1])
        return pd.DataFrame({"r1": [v, v / 2, v / 4]}, index=future_regressor.index[:forecast_length])


def _cfg(tmp_path):
    return RunConfig(dataset_label="synthetic_search", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"],
                     test_start="2026-01-04", root=str(tmp_path), eval_seeds=(1, 2), selection_seed=1,
                     models={"autots_wr": {"window_size": 60}, "autots_mr": {},
                             "autots_search": {"num_validations": 2,
                                               "templates": [{"model": "wr", "window_size": 60, "regressor": "LightGBM"},
                                                             {"model": "mr", "regressor": "xgboost"}]}})


def _write_best(exp, model, ext):
    """Input của `autots-search` giờ là win sau prune + confirmation của chính probe (không còn stage union)."""
    (exp / "wins").mkdir(parents=True, exist_ok=True)
    (exp / "wins" / f"{model}.json").write_text(json.dumps(
        {"model": model, "which": "prune", "colset": {"b0": [], "ext": list(ext)},
         "rmse_mean": [[50.0, 70.0, 90.0]], "e0": [[100.0, 140.0, 170.0]],
         "eps": 0.02, "eval_seeds": [1, 2], "median_gain_vs_e0": 0.1}), encoding="utf-8")


def _prepare(tmp_path, store, folds, monkeypatch, ext_wr, ext_mr):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    _write_best(exp, "autots_wr", ext_wr)
    _write_best(exp, "autots_mr", ext_mr)
    # tiền đề hợp lệ: champion ban đầu = LightGBM (§3) đã có từ `loop --model lgbm`
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "champion.json").write_text(json.dumps(
        {"model": "lgbm", "colset": {"b0": [], "ext": []}, "rmse_mean": [[60.0, 80.0, 100.0]], "eps": 0.02,
         "e0": [[100.0, 140.0, 170.0]]}), encoding="utf-8")
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_store", lambda c, **k: (store, folds, None, None))
    seen = []

    def fake_search(df_tr, R_tr, template, num_validations, seed, autots_cls=None):
        seen.append({"df_last": df_tr.index[-1], "n_bar": len(df_tr), "n_tmpl": len(template),
                     "nv": num_validations, "seed": seed, "R_cols": tuple(R_tr.columns)})
        name = "WindowRegression" if "window_size" in str(template.iloc[0].get("ModelParameters", "")) else "MultivariateRegression"
        return name, {"window_size": 60, "regression_type": "User", "datepart_method": None,
                      "regression_model": {"model": "LightGBM", "model_params": {}}}, pd.DataFrame()

    import p0.autots_search as A

    monkeypatch.setattr(A, "search_best_template", fake_search)
    monkeypatch.setattr(A, "template_frame", lambda specs, seed, **k: pd.DataFrame(
        [{"Model": "WindowRegression" if s.get("model") == "wr" else "MultivariateRegression",
          "ModelParameters": json.dumps({"window_size": 60} if s.get("model") == "wr" else {"holiday": False}),
          "TransformationParameters": "{}", "Ensemble": 0} for s in specs]))
    monkeypatch.setattr(cli, "_autots_probe_model", lambda c, group, allow_cpu, frozen=None: __import__(
        "p0.models_autots", fromlist=["AutoTSModel"]).AutoTSModel(
        kind="mr" if group == "mr" else "wr", device="cpu", allow_cpu=True, tail_bars=150,
        window_size=60 if group == "mr" else int(group.split(":")[1]), model_cls=StubFrozen, frozen=frozen))
    return cfg, exp, seen


# ----------------------------------------------------------------------------- G7: chỉ chạy sau freeze
def test_search_requires_frozen_feature_sets(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        cli.cmd_autots_search(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "loop --model autots_wr" in str(e.value)  # phải có win sau confirmation trước


# ----------------------------------------------------------------------------- G8 + G9 + G10
def test_search_two_sets_independent_outer_clean_and_project_metric(tmp_path, store, folds, monkeypatch):
    cfg, exp, seen = _prepare(tmp_path, store, folds, monkeypatch, ["ret_60"], ["bb_pctb_20", "ret_60"])
    cli.cmd_autots_search(cfg, Namespace(smoke=True, allow_cpu=True))

    # G8: hai frozen set × 2 nhóm shift × mỗi fold → search độc lập, đúng số lần, đúng feature set của từng set
    n_folds, n_groups, n_sets = len(folds), 2, 2
    assert len(seen) == n_sets * n_groups * n_folds
    cols = {s["R_cols"] for s in seen}
    assert cols == {("ret_60",), ("bb_pctb_20", "ret_60")}  # mỗi set giữ đúng regressor của nó, không lẫn sang set kia
    assert {s["nv"] for s in seen} == {2} and {s["seed"] for s in seen} == {cfg.sel_seed}  # search chỉ ở selection_seed

    # G9: outer VAL không bao giờ đi vào search — df kết thúc ở ES, trước purge 60' và trước VAL start
    for s, f in zip(seen, [f for _ in range(n_sets * n_groups) for f in folds]):
        val_start = pd.Timestamp(f.val.start, unit="s")
        assert s["df_last"] <= pd.Timestamp(f.es.end, unit="s") - pd.Timedelta(minutes=1)
        assert s["df_last"] <= val_start - pd.Timedelta(minutes=60)

    # G10: AutoTS-final chọn bằng metric project (không phải điểm nội bộ AutoTS), và chọn Ở SELECTION_SEED
    df = pd.read_csv(exp / "autots_search.csv")
    assert len(df) == n_sets * n_groups
    win = json.loads((exp / "wins" / "autots.json").read_text(encoding="utf-8"))
    assert win["model"] == "autots" and win["role"] == "AutoTS-final"
    assert win["source"] == df.loc[df["MedianGain_vs_E0_sel"].idxmax(), "candidate"]  # chọn theo cột @selection_seed
    assert win["selection_seed"] == cfg.sel_seed and win["eval_seeds"] == list(cfg.eval_seeds)
    # ε của AutoTS-final tính từ CHÍNH 3 bảng RMSE của nó, không mượn ε của probe (probe eps = 0.02)
    from p0.metrics import seed_noise_cells, seed_noise_eps

    tabs = [np.array(t) for t in win["seed_rmse"]]
    assert len(tabs) == len(cfg.eval_seeds)
    assert np.allclose(win["noise_cells"], np.round(seed_noise_cells(tabs), 5), atol=1e-4)
    assert np.isclose(win["eps"], seed_noise_eps(tabs, cfg.eps_floor_pp))
    assert np.allclose(win["rmse_mean"], np.mean(tabs, axis=0))
    assert len(win["templates_per_fold"]) == n_folds  # template freeze theo TỪNG fold
    assert (exp / "wins" / "autots_seed0.npz").exists() and (exp / "wins" / "autots_seed1.npz").exists()

    # champion: AutoTS-final mới là thứ đi so champion (probe thì không)
    ch = pd.read_csv(exp / "champion_log.csv")
    assert (ch["model"] == "autots").any()
    assert len(list((exp / "autots_templates").glob("best_*.json"))) == n_sets * n_groups * n_folds


def test_search_dedups_identical_sets(tmp_path, store, folds, monkeypatch):
    cfg, exp, seen = _prepare(tmp_path, store, folds, monkeypatch, ["ret_60"], ["ret_60"])
    cli.cmd_autots_search(cfg, Namespace(smoke=True, allow_cpu=True))
    assert len(seen) == 1 * 2 * len(folds)  # hai bộ trùng → chỉ search MỘT set
    assert len(pd.read_csv(exp / "autots_search.csv")) == 2  # 2 nhóm shift của cùng một set


def test_probe_model_builder_is_callable_for_each_group(tmp_path):
    """`_autots_probe_model` phải dựng được model thật cho từng nhóm shift (test khác monkeypatch mất hàm này)."""
    cfg = _cfg(tmp_path)
    wr = cli._autots_probe_model(cfg, "wr:60", allow_cpu=True)
    mr = cli._autots_probe_model(cfg, "mr", allow_cpu=True)
    assert (wr.name, wr.kind, wr.window_size) == ("autots_wr", "wr", 60)
    assert (mr.name, mr.kind) == ("autots_mr", "mr")
    assert wr.shift_bars() == 59 and mr.shift_bars() == -1  # căn thời gian regressor theo từng model
    frozen = cli._autots_probe_model(cfg, "wr:30", allow_cpu=True,
                                     frozen=("WindowRegression", {"window_size": 30, "regression_type": "User"}))
    assert frozen.frozen[0] == "WindowRegression" and frozen.window_size == 30 and frozen.use_es is True
