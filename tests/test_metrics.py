import numpy as np

from p0.metrics import cell_metrics, decide, e0_rmse, gain_pp, mean_rmse_over_seeds, price_from_logret, seed_noise_eps, summarize


def test_e0_equals_zero_prediction():
    rng = np.random.default_rng(0)
    c_t = 80_000 + rng.normal(0, 100, 500)
    c_future = c_t[:, None] * np.exp(rng.normal(0, 7.65e-4, (500, 3)) * np.sqrt([1, 2, 3]))
    m = cell_metrics(c_t, c_future, np.zeros((500, 3)))
    assert np.allclose(m["rmse"], e0_rmse(c_t, c_future))
    assert np.allclose(price_from_logret(c_t, np.zeros((500, 3))), c_t[:, None])


def test_perfect_prediction_zero_error():
    rng = np.random.default_rng(1)
    c_t = 80_000 + rng.normal(0, 100, 200)
    y = rng.normal(0, 1e-3, (200, 3))
    c_future = c_t[:, None] * np.exp(y)
    m = cell_metrics(c_t, c_future, y)
    assert np.allclose(m["rmse"], 0, atol=1e-6) and np.allclose(m["r"], 1.0) and np.allclose(m["dir_acc"], 1.0)


def test_gain_and_summary():
    base = np.full((5, 3), 60.0)
    cand = base * (1 - 0.001)  # tốt hơn 0.1% → Gain = 0.1 pp
    g = gain_pp(cand, base)
    assert np.allclose(g, 0.1)
    s = summarize(g)
    assert np.isclose(s["MedianGain"], 0.1) and s["WinRate"] == 1.0 and s["n_cells"] == 15
    assert decide(0.0, 0.02) == "KEEP" and decide(-0.019, 0.02) == "KEEP" and decide(-0.021, 0.02) == "DROP"


def test_mean_over_seeds_and_eps():
    t = [np.full((5, 3), 60.0), np.full((5, 3), 61.0), np.full((5, 3), 62.0)]
    assert np.allclose(mean_rmse_over_seeds(t), 61.0)
    gains = [np.full((5, 3), 0.01), np.full((5, 3), -0.01)]
    assert seed_noise_eps(gains, 0.005) == max(0.005, np.std(np.concatenate([g.ravel() for g in gains])))
