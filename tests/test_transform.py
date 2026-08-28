import numpy as np
import pytest

from p0.transform import TargetTransform


def test_b0_transform_has_inplace_broadcast_bug():
    """Ghi nhận: Baseline_LGBM.TargetTransform.fit raise ValueError (denom (n,1) *= (1,3)) → B0 gốc không chạy nguyên bản."""
    from Baseline_LGBM import TargetTransform as B0Transform

    rng = np.random.default_rng(0)
    y = rng.normal(0, 1e-3, (100, 3)).astype(np.float32)
    rv = np.abs(rng.normal(7e-4, 1e-4, 100)).astype(np.float32)
    with pytest.raises(ValueError):
        B0Transform.fit(y, rv, np.arange(100))


def test_transform_roundtrip_and_train_only():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1e-3, (500, 3)).astype(np.float32)
    rv = np.abs(rng.normal(7e-4, 1e-4, 500)).astype(np.float32)
    train = np.arange(300)
    tr = TargetTransform.fit(y, rv, train)
    assert tr.mean.shape == (3,) and tr.scale.shape == (3,) and tr.volatility_floor > 0
    z = tr.encode(y, rv, train)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4) and np.allclose(z.std(axis=0), 1, atol=1e-3)
    back = tr.decode(z, rv[train])
    assert np.allclose(back, y[train], atol=1e-7)
    # công thức đúng như B0: normalized = y / (max(rv, floor) · sqrt(h))
    denom = np.maximum(rv[train], tr.volatility_floor)[:, None] * np.sqrt(np.array([1, 2, 3], np.float32))[None, :]
    assert np.allclose(tr.mean, (y[train] / denom).mean(axis=0), atol=1e-6)
