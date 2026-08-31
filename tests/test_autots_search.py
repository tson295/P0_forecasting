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
    (exp / "autots_best").mkdir(parents=True, exist_ok=True)
    (exp / "autots_best" / f"{model}.json").write_text(json.dumps(
        {"model": model, "role": "probe_best_feature_set", "chosen": "riêng",
         "colset": {"b0": [], "ext": list(ext)}, "rmse_mean": [[50.0, 70.0, 90.0]], "e0": [[100.0, 140.0, 170.0]],
         "eps": 0.02, "eval_seeds": [1, 2], "median_gain_vs_e0": 0.1}), encoding="utf-8")


def _prepare(tmp_path, store, folds, monkeypatch, ext_wr, ext_mr):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    _write_best(exp, "autots_wr", ext_wr)
    _write_best(exp, "autots_mr", ext_mr)
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
    assert "autots-union" in str(e.value)  # phải freeze feature set trước khi chạy framework


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

    # G10: AutoTS-final chọn bằng metric project (MedianGain vs E0 trên outer VAL), không phải điểm nội bộ AutoTS
    df = pd.read_csv(exp / "autots_search.csv")
    assert len(df) == n_sets * n_groups
    win = json.loads((exp / "wins" / "autots.json").read_text(encoding="utf-8"))
    assert win["model"] == "autots" and win["role"] == "AutoTS-final"
    assert win["source"] == df.loc[df["MedianGain_vs_E0"].idxmax(), "candidate"]
    assert np.isclose(win["median_gain_vs_e0"], df["MedianGain_vs_E0"].max(), atol=1e-3)
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
