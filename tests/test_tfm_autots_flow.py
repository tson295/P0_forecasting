"""Flow feature-selection chung + TimesFM/AutoTS (rev 2026-09-03: một đường TimesFM-LoRA, AutoTS hai probe, không union).

Khoá đúng những gì plan yêu cầu:
- vòng add-one chung: candidate i chạy trên S hiện tại + f_i, KEEP thì candidate sau thấy feature vừa KEEP,
  và model được CHẠY LẠI mỗi khi feature set đổi (không tái dùng prediction cũ);
- confirmation: model có ES dùng fixed rounds ở add-one/prune nhưng `rounds=None` (ES bật lại) ở confirmation;
- TimesFM: chỉ MỘT model `tfm` (LoRA, covariate = ext), không còn `tfm_b0`/`tfm_ext`; covariate không kéo cột B0;
- AutoTS: WR/MR đều là probe, không còn stage union; probe không chạm framework AutoTS.
"""
import ast

import numpy as np
import pytest

from p0 import cli
from p0.harness import ColSet, run_config
from p0.loop import add_one_loop, confirm, prune_pi
from p0.models import make_model
from p0.models_tfm import COVARIATE_SCOPES, TimesFMLoRAModel, TimesFMModel

from test_tfm_autots import StubTFM


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
    calls = m.calls[n0:]
    assert len(calls) == len(cands) * len(folds)
    for i, c in enumerate(cands):
        assert calls[i * len(folds)]["n_col"] == len(base.names) + len(seen[i])
    assert all(c["rounds"] == (5, 5, 5) for c in calls)  # add-one dùng FIXED rounds (ES off)


def test_confirmation_turns_es_back_on(store, folds):
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


# ============================================================================= TimesFM: MỘT đường LoRA
def test_single_timesfm_path_only():
    assert COVARIATE_SCOPES == {"b0star": "all", "ext": "ext"}
    assert set(cli.PROBE_MODELS) == {"autots_wr", "autots_mr", "tfm"} and cli.FINAL_STEP["tfm"] == "tfm-final"
    assert not hasattr(cli, "TFM_BRANCH_BASE")
    m = make_model("tfm", {"device": "cpu"}, allow_cpu=True)
    assert isinstance(m, TimesFMLoRAModel) and (m.name, m.covariate_scope, m.series_covariates) == ("tfm", "ext", "ext")
    assert m.supports_rounds and m.seed_dependent and m.torch_compile is False
    for old in ("tfm_b0", "tfm_ext"):
        with pytest.raises(KeyError):
            make_model(old, {"device": "cpu"}, allow_cpu=True)


def test_timesfm_covariates_are_ext_only_never_b0(store, folds):
    """Xuất phát S = ∅ (native); covariate chỉ là cột ext đang xét; cột B0 không bao giờ thành covariate."""
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope="ext", model=StubTFM())
    st_native = run_config(store, m, ColSet((), ()), folds[:1], rounds=None, seed=1).states[0]
    assert st_native.X_val.cov is None and st_native.X_val.cov_names == ()
    st_c = run_config(store, m, ColSet((), ("ret_60",)), folds[:1], rounds=None, seed=1).states[0]
    assert st_c.X_val.cov_names == ("ret_60",) and st_c.X_val.cov.shape[1] == 1
    st_mix = run_config(store, m, ColSet(store.b0_names[:5], ("ret_60",)), folds[:1], rounds=None, seed=1).states[0]
    assert st_mix.X_val.cov_names == ("ret_60",)


def test_prune_positions_for_series_model_only_new_ext(store, folds):
    """prune PI của model series trỏ đúng cột ext MỚI (cột khoá không bị xét, không bị bỏ)."""
    cs = ColSet((), ("ret_60", "bb_pctb_20"), ("ret_60",))
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope="ext", model=StubTFM())
    pruned, df = prune_pi(store, m, cs, folds[:1], rounds=None, seed=1, repeats=1)
    assert list(df["col"]) == ["bb_pctb_20"] and "ret_60" in pruned.ext and pruned.locked == ("ret_60",)


# ============================================================================= AutoTS: probe, không còn union
def test_no_union_stage_left():
    assert not hasattr(cli, "cmd_autots_union")
    src = cli.Path(cli.__file__).read_text(encoding="utf-8")
    assert "autots-union" not in src and "F_union" not in src and "autots_best" not in src
    for m in ("autots_wr", "autots_mr"):
        assert cli.FINAL_STEP[m] == "autots-search" and m in cli.PROBE_MODELS


def test_probe_models_never_call_autots_framework():
    """Probe chỉ dùng 2 class cố định; framework `AutoTS` nằm ở module riêng `autots_search.py`."""
    src = (cli.Path(cli.__file__).parent / "models_autots.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "AutoTS" not in imported and "AutoTS" not in names
    assert {"WindowRegression", "MultivariateRegression"} <= imported
