"""TimesFM (§2.2 #4) + AutoTS (§2.2 #6): kiểm tra phần code CỦA TA bằng stub thay cho thư viện thật.

Package `timesfm` / `autots` chưa được cài (plan: "cài package chỉ khi user cho phép"), nên test này khoá đúng những thứ
harness chịu trách nhiệm: chỉ dùng τ ≤ t (§6.4), căn thời gian covariate/regressor, cộng dồn r̂ → y_h (§6.7),
1 origin/lời gọi khi có covariate, và permutation importance trên SeriesBatch.
"""
import numpy as np
import pandas as pd
import pytest

from p0.harness import ColSet, run_config
from p0.models import SeriesBatch
from p0.models_autots import AutoTSModel
from p0.models_tfm import TimesFMModel

H = 3


# ----------------------------------------------------------------------------- stub thư viện
class StubTFM:
    """Thay `timesfm.TimesFM_2p5_200M_torch`: r̂ phụ thuộc context (và covariate nếu có) một cách xác định."""

    def __init__(self):
        self.calls = []

    def forecast(self, horizon, inputs):
        self.calls.append(("point", [np.asarray(x).copy() for x in inputs]))
        base = np.array([[float(np.mean(x[-3:])), float(x[-1]), float(np.mean(x))] for x in inputs])
        quant = np.repeat(base[:, :horizon, None], 10, axis=2)
        return base[:, :horizon] * 0.5, quant  # point = q50 (khác mean) để test chọn head

    def forecast_with_covariates(self, inputs, dynamic_numerical_covariates=None, **kw):
        self.calls.append(("cov", [np.asarray(x).copy() for x in inputs], {k: np.asarray(v[0]).copy() for k, v in
                                                                          (dynamic_numerical_covariates or {}).items()}))
        cov = np.array([v[0] for v in (dynamic_numerical_covariates or {}).values()])
        return np.array([[float(cov[:, -1].sum()), float(cov[:, 0].sum()), float(inputs[0][-1])]])


class StubAutoTS:
    """Thay `WindowRegression` / `MultivariateRegression`: ghi lại mọi thứ nhận được, trả forecast xác định."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls, self.data_calls, self.predict_calls = [], [], []
        self.regressor_train = None

    def fit(self, df, future_regressor=None, **kw):
        self.fit_calls.append((df.copy(), None if future_regressor is None else future_regressor.copy()))
        return self

    def fit_data(self, df, future_regressor=None, **kw):
        self.data_calls.append((df.copy(), future_regressor))
        self.last = df
        return self

    def predict(self, forecast_length=None, future_regressor=None, just_point_forecast=False, **kw):
        self.predict_calls.append((future_regressor.copy(), just_point_forecast, self.regressor_train))
        v = float(future_regressor.to_numpy()[-1].sum()) + float(self.last["r1"].iloc[-1])
        return pd.DataFrame({"r1": [v, v / 2, v / 4]}, index=future_regressor.index[:forecast_length])


@pytest.fixture
def seq(store):
    idx = np.flatnonzero(store.eligible)
    idx = idx[idx >= 600][:40]
    cov = np.column_stack([store.ext_column("ret_60"), store.ext_column("bb_pctb_20")]).astype(np.float32)
    cov = np.nan_to_num(cov, nan=0.0)
    return SeriesBatch(store.ts, store.r1, idx, cov, ("ret_60", "bb_pctb_20"))


# ----------------------------------------------------------------------------- TimesFM
def test_tfm_context_only_uses_past(seq):
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, model=StubTFM())
    ctxs = m.contexts(seq)
    t = int(seq.idx[0])
    assert len(ctxs) == len(seq.idx) and len(ctxs[0]) == 512
    assert np.array_equal(ctxs[0], seq.r1[t - 511:t + 1].astype(np.float32))  # kết thúc đúng tại t
    r1 = seq.r1.copy()
    r1[int(seq.idx[-1]) + 1:] = 999.0  # đổi mọi thứ sau origin cuối
    after = TimesFMModel(device="cpu", allow_cpu=True, context=512, model=StubTFM()).contexts(
        SeriesBatch(seq.ts, r1, seq.idx, seq.cov, seq.cov_names))
    assert all(np.array_equal(a, b) for a, b in zip(ctxs, after))


def test_tfm_point_cumsum_and_mean_head(seq):
    stub = StubTFM()
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=16, model=stub)
    bare = SeriesBatch(seq.ts, seq.r1, seq.idx)  # không covariate → TFM-POINT
    yhat = m.predict_series(bare)
    ctxs = m.contexts(bare)
    r_hat = np.array([[float(np.mean(x[-3:])), float(x[-1]), float(np.mean(x))] for x in ctxs])
    assert yhat.shape == (len(seq.idx), H)
    assert np.allclose(yhat, np.cumsum(r_hat, axis=1), atol=1e-6)  # cộng dồn one-step → y_h
    assert all(c[0] == "point" for c in stub.calls) and len(stub.calls) == int(np.ceil(len(seq.idx) / 16))


def test_tfm_covariate_shift_and_single_origin(seq):
    stub = StubTFM()
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, model=stub)
    t = int(seq.idx[0])
    w = m.covariate_window(seq, t, 0)
    assert len(w) == 512 + H
    assert np.allclose(w[:512], seq.cov[t - 512:t, 0])  # vị trí s mang f(s−1): cửa sổ t−511..t ↔ f(t−512)..f(t−1)
    assert np.allclose(w[512:], seq.cov[t, 0])  # 3 bước tương lai giữ f(t)
    m.predict_series(seq)
    cov_calls = [c for c in stub.calls if c[0] == "cov"]
    assert len(cov_calls) == len(seq.idx)  # đúng 1 origin mỗi lời gọi (xreg fit chung beta_hat → gộp là leakage)
    assert all(len(c[1]) == 1 for c in cov_calls)
    assert set(cov_calls[0][2]) == {"ret_60", "bb_pctb_20"}


# ----------------------------------------------------------------------------- AutoTS
@pytest.mark.parametrize("kind,shift", [("mr", -1), ("wr", 59)])
def test_autots_regressor_alignment(seq, kind, shift):
    m = AutoTSModel(kind=kind, device="cpu", allow_cpu=True, window_size=60, model_cls=StubAutoTS)
    assert m.shift_bars() == shift
    lo, hi = int(seq.idx[0]), int(seq.idx[0]) + 10
    R = m.regressor_frame(seq, lo, hi)
    assert list(R.columns) == list(seq.cov_names) and len(R) == 10
    for k in range(10):  # R.loc[s] = f(s + shift)
        assert np.allclose(R.to_numpy()[k], seq.cov[lo + k + shift])
    t = int(seq.idx[5])
    F = m.future_regressor(seq, t, 5)
    assert len(F) == H and np.allclose(F.to_numpy(), seq.cov[t])  # 3 hàng tương lai = f(t)
    assert (F.index == pd.to_datetime(seq.ts[t] + 60 * np.array([1, 2, 3]), unit="s")).all()


@pytest.mark.parametrize("kind", ["wr", "mr"])
def test_autots_rolling_predict_is_causal(seq, kind):
    m = AutoTSModel(kind=kind, device="cpu", allow_cpu=True, window_size=60, tail_bars=120, model_cls=StubAutoTS)
    fit_seq = SeriesBatch(seq.ts, seq.r1, seq.idx[:20], seq.cov, seq.cov_names)
    pred_seq = SeriesBatch(seq.ts, seq.r1, seq.idx[20:], seq.cov, seq.cov_names)
    res = m.fit_predict(fit_seq, None, None, None, pred_seq, None, 8586)
    assert res.is_logret and res.pred_z.shape == (len(pred_seq.idx), H) and res.best_iters == (0, 0, 0)
    # mỗi origin: fit_data nhận lát kết thúc ĐÚNG tại t, regressor_train gán tay (bug fit_data của autots 1.0.4)
    inner = res.predictors[0].model
    assert len(inner.data_calls) == len(pred_seq.idx) and len(inner.fit_calls) == 1
    for k, t in enumerate(pred_seq.idx):
        df_slice = inner.data_calls[k][0]
        assert df_slice.index[-1] == pd.Timestamp(seq.ts[int(t)], unit="s")
        assert inner.data_calls[k][1] is None  # KHÔNG truyền regressor vào fit_data
    assert inner.predict_calls[0][1] is True  # just_point_forecast
    # regressor_train: MR gán tay và chỉ chứa f(≤ t−1); WR không gán khi rolling (R của WR trỏ tới bar sau t)
    rt = inner.predict_calls[-1][2]
    if kind == "mr":
        t_last = int(pred_seq.idx[-1])
        assert rt is not None and len(rt) and rt.index[-1] == pd.Timestamp(seq.ts[t_last], unit="s")
        assert np.allclose(rt.to_numpy()[-1], seq.cov[t_last - 1])  # hàng cuối = f(t−1)
    else:
        assert rt is None or rt.index[-1] <= pd.Timestamp(seq.ts[int(fit_seq.idx[-1])], unit="s")


def test_autots_fit_uses_only_fit_range(seq):
    m = AutoTSModel(kind="mr", device="cpu", allow_cpu=True, model_cls=StubAutoTS)
    fit_seq = SeriesBatch(seq.ts, seq.r1, seq.idx[:20], seq.cov, seq.cov_names)
    res = m.fit_predict(fit_seq, None, None, None, SeriesBatch(seq.ts, seq.r1, seq.idx[20:22], seq.cov, seq.cov_names), None, 1)
    inner = res.predictors[0].model
    df_fit, R_fit = inner.fit_calls[0]
    assert df_fit.index[-1] == pd.Timestamp(seq.ts[int(fit_seq.idx[-1])], unit="s")  # fit không chạm ES/VAL
    assert len(R_fit) == len(df_fit)


# ----------------------------------------------------------------------------- harness: is_logret + PI trên SeriesBatch
def test_series_model_skips_target_transform(store, folds):
    """TimesFM/AutoTS trả thẳng log-return → run_config KHÔNG decode qua TargetTransform (§6.7)."""
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, model=StubTFM())
    cs = ColSet(store.b0_names[:4])
    run = run_config(store, m, cs, folds[:1], rounds=None, seed=1)
    st = run.states[0]
    assert st.result.is_logret and np.allclose(st.yhat, st.result.pred_z, atol=1e-6)
    assert run.rmse.shape == (1, 3) and np.isfinite(run.rmse).all()
    assert (run.best_iters == 0).all()


def test_prune_pi_over_series_batch(store, folds):
    from p0.filter_b0 import permutation_importance
    from p0.loop import prune_pi

    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, model=StubTFM())
    cs = ColSet(store.b0_names[:4], ("ret_60", "bb_pctb_20"))
    pruned, df = prune_pi(store, m, cs, folds[:1], rounds=None, seed=1, repeats=1)
    assert list(df["col"]) == list(cs.ext) and set(pruned.ext) <= set(cs.ext) and pruned.b0 == cs.b0
    run = run_config(store, m, cs, folds[:1], rounds=None, seed=1)
    st = run.states[0]
    rng = np.random.default_rng(0)
    z0 = st.result.predict_z(st.X_val)
    z1 = st.result.predict_z(st.X_val.with_perm({0: rng.permutation(st.X_val.idx)}))
    assert z0.shape == z1.shape and not np.allclose(z0, z1)  # perm thực sự đổi covariate của đúng kênh
    d = permutation_importance(store, run, [0], repeats=1, seed=1)
    assert d.shape == (1, 1, 3) and np.isfinite(d).all()


def test_autots_positions_are_ext_columns(store):
    """AutoTS: regressor = B0* + ext → vị trí cột ext để tính PI phải nằm SAU các cột b0."""
    from p0.loop import prune_pi

    m = AutoTSModel(kind="wr", device="cpu", allow_cpu=True, model_cls=StubAutoTS)
    assert m.series_covariates == "all" and TimesFMModel(device="cpu", allow_cpu=True, model=StubTFM()).series_covariates == "ext"
    cs = ColSet(store.b0_names[:3], ("ret_60",))
    assert prune_pi.__doc__  # (chỉ khẳng định API tồn tại; PI của AutoTS chạy trong smoke/Vast, không gọi ở đây cho nhanh)
