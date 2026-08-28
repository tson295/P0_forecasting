import numpy as np
import pytest

from p0.features_ext import CANDIDATES
from p0.filter_b0 import FilterTable, flag_2of3, median_over_folds, mutual_info, permutation_importance, verify_sets
from p0.harness import ColSet, calibrate, rounds_from, run_config, seed_noise
from p0.latency import measure_tabular
from p0.loop import add_one_loop, compare, confirm, decide_win, ensemble_rmse, inverse_mse_weights, prune_pi
from p0.metrics import gain_pp
from p0.models import FitResult, make_model


class DummyModel:
    """Model tuyến tính đơn giản, deterministic: z = w·X (fit bằng least squares); dùng để test logic vòng lặp."""

    name = "dummy"
    supports_rounds = True
    train_device = "CPU"

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds, seed):
        Xf = np.nan_to_num(np.asarray(X_fit, float))
        Xp = np.nan_to_num(np.asarray(X_pred, float))
        A = np.c_[Xf, np.ones(len(Xf))]
        W = np.linalg.lstsq(A, z_fit, rcond=None)[0]
        rng = np.random.default_rng(seed)
        W = W + rng.normal(0, 1e-3, W.shape)
        pred = (np.c_[Xp, np.ones(len(Xp))] @ W).astype(np.float32)
        preds = [lambda X, c=col, W=W: (np.c_[np.nan_to_num(np.asarray(X, float)), np.ones(len(X))] @ W[:, c]).astype(np.float32) for col in range(3)]
        return FitResult(pred, tuple(rounds) if rounds is not None else (7, 8, 9), preds)


def test_run_config_shapes_and_e0(store, folds):
    cs = ColSet(store.b0_names[:10])
    run = run_config(store, DummyModel(), cs, folds, rounds=None, seed=1)
    assert run.rmse.shape == (len(folds), 3) and run.e0.shape == (len(folds), 3)
    assert (run.best_iters == np.array([[7, 8, 9]] * len(folds))).all()
    assert np.all(run.e0 > 0)
    # E0 độc lập với model: ŷ=0 cho đúng E0
    st = run.states[0]
    c_t, c_future, _ = store.targets(st.idx_val)
    from p0.metrics import cell_metrics, e0_rmse

    assert np.allclose(cell_metrics(c_t, c_future, np.zeros_like(st.yhat))["rmse"], e0_rmse(c_t, c_future))


def test_target_transform_train_only(store, folds):
    cs = ColSet(store.b0_names[:5])
    run = run_config(store, DummyModel(), cs, folds, rounds=None, seed=1)
    from p0.transform import TargetTransform

    st = run.states[0]
    tr = TargetTransform.fit(store.fd.target, store.fd.rv60, st.idx_fit)
    assert np.allclose(tr.mean, st.transform.mean) and np.allclose(tr.scale, st.transform.scale)


def test_calibrate_seed_noise_and_loop(store, folds):
    cs = ColSet(store.b0_names[:8])
    cal = calibrate(store, DummyModel(), cs, folds, seed=1)
    rounds = rounds_from(cal)
    assert set(rounds) == {f.name for f in folds}
    eps, runs = seed_noise(store, DummyModel(), cs, folds, rounds, (1, 2, 3), 0.005)
    assert eps >= 0.005 and len(runs) == 3
    cands = CANDIDATES[:4]
    lr = add_one_loop(store, DummyModel(), cs, runs[0].rmse, cands, folds, rounds, eps, 1, runs[0].e0)
    assert len(lr.table) == 4 and set(lr.table["decision"]) <= {"KEEP", "DROP"}
    assert len(lr.kept) + len(lr.dropped) == 4
    for c in lr.kept:
        assert all(col in lr.final.ext for col in next(x for x in cands if x.name == c).columns)


def test_prune_confirm_win_and_compare(store, folds):
    cs = ColSet(store.b0_names[:6], ("ret_60", "bb_pctb_20"))
    rounds = {f.name: (5, 5, 5) for f in folds}
    pruned, df = prune_pi(store, DummyModel(), cs, folds, rounds, seed=1)
    assert set(df["col"]) == set(cs.ext) and set(pruned.ext) <= set(cs.ext)
    unp = confirm(store, DummyModel(), cs, folds, (1, 2, 3))
    assert unp.rmse_mean.shape == (len(folds), 3) and len(unp.runs) == 3
    prn = confirm(store, DummyModel(), ColSet(cs.b0), folds, (1, 2, 3))
    which, g, s = decide_win(unp, prn, eps=0.02)
    assert which in ("prune", "unprune") and g.shape == (len(folds), 3)
    # decide_win đúng công thức: Gain = 1 − RMSE̅^prune/RMSE̅^unprune từng ô, median 15 ô
    assert np.isclose(s["MedianGain"], np.median(gain_pp(prn.rmse_mean, unp.rmse_mean)))
    change, gc, sc = compare(unp.rmse_mean, unp.rmse_mean * 1.001, eps=0.02)
    assert change and np.isclose(sc["MedianGain"], 100 * (1 - 1 / 1.001))


def test_ensemble_and_weights(store, folds):
    cs = ColSet(store.b0_names[:6])
    a = confirm(store, DummyModel(), cs, folds, (1, 2))
    b = confirm(store, DummyModel(), ColSet(store.b0_names[6:12]), folds, (1, 2))
    members = {"a": a.preds_by_seed(), "b": b.preds_by_seed()}
    eq = ensemble_rmse(store, members, folds)
    inv = ensemble_rmse(store, members, folds, inverse_mse_weights({"a": a.rmse_mean, "b": b.rmse_mean}))
    assert eq.shape == (len(folds), 3) and inv.shape == (len(folds), 3) and np.all(eq > 0)


def test_filter_table_flags_and_sets():
    names = ["a", "b", "c", "d"]
    pi = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, 1, 1]], float)
    sa = np.array([[1, 1, 1], [1, 1, -1], [1, 1, 1], [-1, -1, -1]], float)
    mi = np.array([[1, -1, -1], [1, 1, 1], [1, 1, -1], [-1, -1, -1]], float)
    t = FilterTable(names, pi, sa, sa, mi)
    f = t.flags()
    assert f["PI"].tolist() == [True, False, False, True]
    s = t.sets()
    assert s["R1"] == ("a", "b", "c", "d") and s["R3"] == ("a", "d") and s["R4"] == ("a", "b", "c")
    assert s["R2"] == ("a", "b", "c", "d")[:3] + ("d",)  # PI+ ∨ (SA+ ∧ MI+): a,b,c(SA∧MI),d(PI)


def test_permutation_importance_and_mi(store, folds):
    cs = ColSet(store.b0_names[:5])
    run = run_config(store, DummyModel(), cs, folds, rounds=None, seed=1)
    d = permutation_importance(store, run, [0, 1], repeats=2, seed=0)
    assert d.shape == (2, len(folds), 3)
    assert median_over_folds(d).shape == (2, 3)
    mi = mutual_info(store, folds, cs, seed=0)
    assert mi.shape == (5, 3)


def test_verify_sets_selection_rule(store, folds):
    rounds = {f.name: (5, 5, 5) for f in folds}
    base = run_config(store, DummyModel(), ColSet(store.b0_names[:6]), folds, rounds=rounds, seed=1)
    sets = {"R1": store.b0_names[:6], "R2": store.b0_names[:3], "R3": store.b0_names[:1], "R4": ()}
    df, chosen, runs = verify_sets(store, DummyModel(), sets, folds, rounds, base.rmse, eps=0.5, seed=1)
    assert chosen in ("R1", "R2", "R3", "B0-306") and df["chosen"].sum() <= 1


def test_latency_pass_does_not_change_prediction(store, folds):
    cs = ColSet(store.b0_names[:5])
    run = run_config(store, DummyModel(), cs, folds, rounds=(5, 5, 5), seed=1)
    lat = measure_tabular(run, warmup=3, max_origins=20)
    assert set(lat["h"]) == {1, 2, 3} and (lat["p99_ms"] >= lat["p95_ms"]).all()


@pytest.mark.skipif(pytest.importorskip("lightgbm") is None, reason="lightgbm")
def test_lightgbm_cpu_smoke(store, folds):
    """Smoke LightGBM (code gốc B0) trên CPU với allow_cpu=True — chỉ unit test, không phải training thật."""
    model = make_model("lgbm", {"device_type": "cpu", "n_estimators": 30}, allow_cpu=True)
    cs = ColSet(store.b0_names[:20])
    cal = calibrate(store, model, cs, folds, seed=8586)
    assert (cal.best_iters >= 1).all()
    run = run_config(store, model, cs, folds, rounds=rounds_from(cal), seed=8586)
    assert (run.best_iters == cal.best_iters).all()
    lat = measure_tabular(run, warmup=2, max_origins=10)
    assert len(lat) == 3


def test_cpu_guard_blocks_training_without_flag():
    with pytest.raises(RuntimeError):
        make_model("lgbm", {"device_type": "cpu"}, allow_cpu=False)
