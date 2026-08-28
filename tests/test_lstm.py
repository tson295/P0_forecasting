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
