r"""AutoTS (§2.2 #6) — 2 model CỐ ĐỊNH, không search. Theo `docs/reference/audit_autots.md` (autots 1.0.4).

- Gọi thẳng `autots.models.sklearn.{WindowRegression, MultivariateRegression}`; KHÔNG dùng `AutoTS(...)`
  (mặc định của nó là search template + transformer + validation/metric riêng — plan cấm).
- Target = `r1` (1 series, wide df, freq 'min'), `forecast_length = 3` → **cộng dồn** → `y_h` (`is_logret=True`).
- Regressor = B0\* + candidate (`regression_type='User'`), **dịch thời gian theo từng model** để chỉ dùng dữ liệu ≤ s−1:
  · MultivariateRegression ghép regressor tại đúng thời điểm target → `R.loc[s] = f(s−1)`
  · WindowRegression ghép tại vị trí ĐẦU cửa sổ → `R.loc[s] = f(s + window_size − 1)`; predict dùng `future_regressor.tail(1)` = `f(t)`
  Ba hàng tương lai (t+1..t+3) luôn giữ giá trị tại t (plan §2.2).
- Rolling-origin KHÔNG refit: `fit` một lần mỗi fold, rồi `fit_data(df ≤ t)` + `predict(forecast_length=3)` cho từng origin.
- Vá bug autots 1.0.4 `sklearn.py:3337` (`future_regressor.reindex(df)` với df là DataFrame → ValueError): gọi `fit_data(df_slice)`
  KHÔNG truyền regressor rồi tự gán `m.regressor_train` (đúng ngữ nghĩa của `fit`), đồng thời cắt đuôi để không concat cả FIT mỗi origin.
- Không sửa thư viện; `n_jobs=1` (tránh nhiều process tranh GPU); `max_windows` lớn (mặc định 5000 cắt mất phần lớn FIT).

Giai đoạn (iii) — **bake-off template GPU** (`search_best_template`, plan §2.2 #6, audit §12.4d phương án A / §12.10):
`AutoTS(initial_template=<DataFrame do ta khai báo, mọi dòng ép GPU>, max_generations=0, transformer_max_depth=0)`
chỉ chạy đúng các dòng ta nạp + validation nội bộ của AutoTS trên **training-side của fold** rồi chọn best theo RMSE.
KHÔNG dùng genetic search: `generate_regressor_params` không sinh khoá `device`/`device_type` và mọi generation ≥ 1
ghi đè params GPU ⇒ sẽ train CPU, vi phạm invariant §0. Template thắng được FREEZE và chạy lại bằng `ModelMonster`
(`AutoTSModel(frozen=...)`) theo đúng vòng rolling ở trên.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS
from .models import FitResult, SeriesBatch, _cpu_guard

TAIL_BARS = 400  # số bar cuối truyền vào fit_data mỗi origin (đủ cho last_window của WR và min_threshold ≥ 90 của MR)

# LƯU Ý (canary trên package thật, 2026-08-31): `autots.models.sklearn.retrieve_regressor` (sklearn.py:471–488)
# đã tự truyền `verbose=-1, random_state=..., n_jobs=...` cho LGBMRegressor rồi `**model_params`, nên KHÔNG được
# đặt các khoá đó ở đây (trùng kwargs → TypeError: got multiple values for keyword argument 'verbose'; AutoTS nuốt
# lỗi này thành "Template Eval Error" và âm thầm bỏ template). Verbosity không phải tham số mô hình.
WR_PARAMS = {"model": "LightGBM", "model_params": {"device_type": "gpu", "n_estimators": 400, "learning_rate": 0.03, "max_depth": 6,
                                                   "num_leaves": 31}}
MR_PARAMS = {"model": "xgboost", "model_params": {"device": "cuda", "tree_method": "hist", "n_estimators": 400, "learning_rate": 0.03,
                                                  "max_depth": 6}}


def _frame(ts: np.ndarray, values: np.ndarray, columns) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values), index=pd.to_datetime(np.asarray(ts), unit="s", utc=True).tz_localize(None), columns=list(columns))


class AutoTSModel:
    """kind = 'wr' (WindowRegression + LightGBM GPU) | 'mr' (MultivariateRegression + XGBoost GPU)."""

    input_kind = "series"
    series_covariates = "all"  # regressor = B0* + ext (plan: "base regressor = B0*")
    supports_rounds = False  # số vòng của regression_model cố định trong config (§1.3)
    lib = "autots"

    def __init__(self, kind: str = "wr", device: str = "cuda", allow_cpu: bool = False, window_size: int = 60,
                 regression_model: dict | None = None, max_windows: int = 200_000, tail_bars: int = TAIL_BARS,
                 n_jobs: int = 1, frequency: str = "min", model_cls=None, frozen: tuple[str, dict] | None = None):
        # frozen = (tên model AutoTS, params đã search) → chạy lại nguyên trạng bằng ModelMonster (giai đoạn iii)
        self.frozen = None if frozen is None else (str(frozen[0]), dict(frozen[1]))
        self.use_es = frozen is not None  # template freeze: refit trên đúng dữ liệu bake-off (FIT+ES)
        if self.frozen is not None:
            kind = "wr" if self.frozen[0] == "WindowRegression" else "mr"
            window_size = int(self.frozen[1].get("window_size", window_size))
        if kind not in ("wr", "mr"):
            raise KeyError(f"AutoTS kind phải là 'wr' hoặc 'mr': {kind}")
        _cpu_guard(device == "cuda", allow_cpu, f"AutoTS-{kind.upper()}")
        self.kind = kind
        self.name = f"autots_{kind}"
        self.window_size, self.max_windows, self.tail_bars = int(window_size), int(max_windows), int(tail_bars)
        self.n_jobs, self.frequency, self._cls = int(n_jobs), frequency, model_cls
        base = dict(WR_PARAMS if kind == "wr" else MR_PARAMS)
        if regression_model is not None:
            base = regression_model
        elif device != "cuda":  # chỉ unit test CPU
            base = {"model": base["model"], "model_params": {k: v for k, v in base["model_params"].items()
                                                             if k not in ("device_type", "device")}}
            base["model_params"]["n_estimators"] = 20
        self.regression_model = base
        self.train_device = "GPU" if device == "cuda" else "CPU"
        self.predict_device = "CPU" if kind == "wr" else self.train_device  # LightGBM predict luôn CPU (đặc tính thư viện)

    # ------------------------------------------------------------------ regressor (căn thời gian, §5 audit)
    def shift_bars(self) -> int:
        """Số bar dịch khi dựng R: MR lấy f(s−1) → +(-1); WR lấy f(s + window_size − 1)."""
        return -1 if self.kind == "mr" else self.window_size - 1

    def regressor_frame(self, seq: SeriesBatch, lo: int, hi: int) -> pd.DataFrame:
        """R trên các bar [lo, hi): R.loc[s] = f(s + shift). NaN ở biên (đã fill 0 khi chuẩn hoá train-only)."""
        sh = self.shift_bars()
        src = np.clip(np.arange(lo, hi) + sh, 0, len(seq.r1) - 1)
        return _frame(seq.ts[lo:hi], seq.cov[src], seq.cov_names)

    def future_regressor(self, seq: SeriesBatch, t: int, k: int) -> pd.DataFrame:
        """3 hàng t+1..t+3, giá trị = f(t) (plan: giữ giá trị tại t). `perm` (PI) thay f_j(t) bằng f_j(t') của origin khác."""
        row = np.asarray(seq.cov[t], dtype=np.float64).copy()
        if seq.perm:
            for j, alt in seq.perm.items():
                row[j] = seq.cov[int(np.asarray(alt)[k]), j]
        ts_future = seq.ts[t] + 60 * np.asarray(HORIZONS, np.int64)
        return _frame(ts_future, np.repeat(row[None, :], len(HORIZONS), axis=0), seq.cov_names)

    # ------------------------------------------------------------------ model
    def _make(self, seed: int):
        if self.frozen is not None:  # template đã freeze: dựng lại đúng model + params mà bake-off đã chọn
            name, params = self.frozen
            if self._cls is not None:
                return self._cls(name, parameters=params, frequency=self.frequency, forecast_length=len(HORIZONS),
                                 prediction_interval=0.9, holiday_country=None, random_seed=seed, verbose=0, n_jobs=self.n_jobs)
            from autots.evaluator.auto_model import ModelMonster

            return ModelMonster(name, parameters=params, frequency=self.frequency, forecast_length=len(HORIZONS),
                                prediction_interval=0.9, holiday_country=None, random_seed=seed, verbose=0, n_jobs=self.n_jobs)
        if self._cls is not None:  # stub trong unit test
            cls = self._cls
        elif self.kind == "wr":
            from autots.models.sklearn import WindowRegression as cls
        else:
            from autots.models.sklearn import MultivariateRegression as cls
        rm = {"model": self.regression_model["model"], "model_params": dict(self.regression_model["model_params"])}
        if str(rm["model"]).lower() in ("xgboost", "xgbregressor"):
            rm["model_params"].setdefault("random_state", seed)  # nhánh xgboost của AutoTS không tự set seed
        common = dict(forecast_length=len(HORIZONS), regression_type="User", regression_model=rm, n_jobs=self.n_jobs,
                      random_seed=seed, frequency=self.frequency, verbose=0)
        if self.kind == "wr":
            return cls(window_size=self.window_size, output_dim="forecast_length", max_windows=self.max_windows,
                       normalize_window=False, scale=False, datepart_method=None, shuffle=False, **common)
        return cls(datepart_method=None, **common)

    def frames(self, seq: SeriesBatch, lo: int, hi: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(df 1 cột 'r1', R regressor đã dịch) trên các bar [lo, hi) — dùng cho fit và cho bake-off."""
        return _frame(seq.ts[lo:hi], seq.r1[lo:hi, None], ["r1"]), self.regressor_frame(seq, lo, hi)

    def fit_range(self, X_fit: SeriesBatch, X_es: SeriesBatch | None) -> tuple[int, int]:
        """Khoảng bar để fit: probe = FIT; template đã freeze (`use_es`) = FIT+ES — đúng bằng dữ liệu bake-off đã dùng."""
        lo, hi = int(X_fit.idx.min()), int(X_fit.idx.max()) + 1
        if self.use_es and X_es is not None and len(getattr(X_es, "idx", ())):
            hi = max(hi, int(X_es.idx.max()) + 1)
        return lo, hi

    def fit_predict(self, X_fit: SeriesBatch, z_fit, X_es, z_es, X_pred: SeriesBatch, rounds, seed: int) -> FitResult:
        lo, hi = self.fit_range(X_fit, X_es)  # lát liên tục trên lưới (đã kiểm không gap §1.1), kết thúc trước purge
        df_fit, R_fit = self.frames(X_fit, lo, hi)
        m = self._make(seed)
        m.fit(df_fit, future_regressor=R_fit)
        predictor = self._make_predictor(m)
        return FitResult(predictor(X_pred), (0, 0, 0), [predictor], is_logret=True)

    def _make_predictor(self, m):
        def predict(seq: SeriesBatch) -> np.ndarray:
            out = np.empty((len(seq.idx), len(HORIZONS)), dtype=np.float64)
            for k, t in enumerate(seq.idx):
                t = int(t)
                a = max(0, t + 1 - self.tail_bars)
                df_slice = _frame(seq.ts[a:t + 1], seq.r1[a:t + 1, None], ["r1"])  # chỉ τ ≤ t
                m.fit_data(df_slice)  # KHÔNG refit; không truyền regressor (bug autots 1.0.4 sklearn.py:3337)
                if self.kind == "mr":
                    # MR.predict nối regressor_train với future_regressor → phải gán tay (đúng ngữ nghĩa fit()), chỉ phần đuôi.
                    # Shift của MR là −1 nên mọi hàng chỉ chứa f(≤ t−1). WR KHÔNG gán: predict của nó chỉ dùng
                    # future_regressor.tail(1), và R của WR (shift +W−1) sẽ trỏ tới bar sau t → không được đưa vào lúc predict.
                    m.regressor_train = self.regressor_frame(seq, a, t + 1)
                fc = m.predict(forecast_length=len(HORIZONS), future_regressor=self.future_regressor(seq, t, k),
                               just_point_forecast=True)
                out[k] = np.asarray(fc).reshape(len(HORIZONS), -1)[:, 0]
            return np.cumsum(out, axis=1).astype(np.float32)  # one-step r̂ → y_h (§6.7)

        predict.model = m  # model AutoTS đã fit của fold này (debug/kiểm tra)
        return predict

