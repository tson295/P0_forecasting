import numpy as np
import pytest

torch = pytest.importorskip("torch")

from p0.harness import ColSet, run_config  # noqa: E402
from p0.models import make_model  # noqa: E402


def test_lstm_cpu_tiny(store, folds):
    model = make_model("lstm", {"device": "cpu", "context": 16, "hidden": 8, "max_epochs": 2, "batch_size": 64}, allow_cpu=True)
    cs = ColSet(tuple(n for n in store.b0_names[:22]), ("ret_60",))
    run = run_config(store, model, cs, folds[:1], rounds=None, seed=1)
    assert run.rmse.shape == (1, 3) and np.isfinite(run.rmse).all()
    fixed = run_config(store, model, cs, folds[:1], rounds=(1, 1, 1), seed=1)
    assert (fixed.best_iters == 1).all()


def test_lstm_prune_pi_over_seqbatch(store, folds):
    """§2.1a: prune PI phải chạy đúng với LSTM (input SeqBatch) — xáo đúng KÊNH ext của sequence, không đụng kênh khác."""
    from p0.filter_b0 import permutation_importance
    from p0.loop import prune_pi

    model = make_model("lstm", {"device": "cpu", "context": 8, "hidden": 6, "max_epochs": 1, "batch_size": 128}, allow_cpu=True)
    cs = ColSet(tuple(n for n in store.b0_names[:22]), ("ret_60", "bb_pctb_20"))
    # vị trí kênh ext trong ma trận LSTM khác vị trí cột trong ma trận tabular
    names = store.fine_names(cs)
    assert names[-2:] == ["ret_60", "bb_pctb_20"] and "rv60" in names

    pruned, df = prune_pi(store, model, cs, folds[:1], rounds=(1, 1, 1), seed=1, repeats=1)
    assert list(df["col"]) == list(cs.ext) and set(pruned.ext) <= set(cs.ext) and pruned.b0 == cs.b0

    # perm thực sự đổi prediction của đúng kênh đó (nếu không, PI luôn = 0 và prune sai)
    run = run_config(store, model, cs, folds[:1], rounds=(1, 1, 1), seed=1)
    st = run.states[0]
    j = names.index("ret_60")
    rng = np.random.default_rng(0)
    z0 = st.result.predict_z(st.X_val)
    z1 = st.result.predict_z(st.X_val.with_perm({j: rng.permutation(st.X_val.idx)}))
    assert z0.shape == z1.shape == (len(st.idx_val), 3) and not np.allclose(z0, z1)
    d = permutation_importance(store, run, [j], repeats=1, seed=1)
    assert d.shape == (1, 1, 3) and np.isfinite(d).all()
