import numpy as np
import pandas as pd
import pytest

from p0.data import grid_frame
from p0.features_ext import ALL_EXT_COLUMNS, CANDIDATES, compute_ext, psar


def test_all_candidates_computed(store):
    df = compute_ext(store.grid, store.raw_lf)
    assert list(df.columns) == list(ALL_EXT_COLUMNS)
    assert len(df) == len(store.grid)
    # sau warmup phải có giá trị hữu hạn ở phần lớn cột
    tail = df.iloc[-500:]
    frac = np.isfinite(tail.to_numpy()).mean(axis=0)
    assert (frac > 0.95).all(), dict(zip(df.columns, frac))


@pytest.mark.parametrize("col", ALL_EXT_COLUMNS)
def test_causality_truncation(store, col):
    """Feature tại t tính trên chuỗi cắt tại t phải bằng chuỗi đầy đủ tại t (không dùng bar > t)."""
    g = store.grid
    n = len(g)
    full = compute_ext(g, store.raw_lf, columns=(col,))[col].to_numpy()
    for cut in (n - 1, n - 7, n - 61):
        gt = g.iloc[: cut + 1]
        lf = store.raw_lf[store.raw_lf["timestamp"] <= g["timestamp"].iloc[cut]] if store.raw_lf is not None else None
        part = compute_ext(gt, lf, columns=(col,))[col].to_numpy()
        a, b = full[cut], part[cut]
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-5, atol=1e-7), (col, cut, a, b)


def test_lookbacks_produce_nan_only_at_warmup(store):
    df = compute_ext(store.grid, store.raw_lf, columns=("ret_1440", "log_rv60_med2d"))
    assert df["ret_1440"].iloc[:1440].isna().all() and np.isfinite(df["ret_1440"].iloc[1440:]).all()
    assert df["log_rv60_med2d"].iloc[:2900].isna().all()


def test_psar_causal_and_flips():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 500)))
    h, l = c * 1.0005, c * 0.9995
    d, s, a = psar(h, l, c)
    assert np.isnan(d[0]) and np.isfinite(d[1:]).all()
    assert set(np.unique(d[1:])) <= {-1.0, 1.0}
    d2, s2, a2 = psar(h[:300], l[:300], c[:300])
    assert np.allclose(s2[1:], s[1:300])  # cắt chuỗi không đổi quá khứ


def test_amount_vwap_is_true_vwap(store):
    df = compute_ext(store.grid, None, columns=("vwap_amt_gap_1",))
    g = store.grid
    expected = np.log(g["close"] / (g["amount"] / g["volume"]))
    ok = np.isfinite(expected)
    assert np.allclose(df["vwap_amt_gap_1"].to_numpy()[ok], expected.to_numpy()[ok], rtol=1e-5)
