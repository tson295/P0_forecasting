"""Adapter model dạng bảng (§2.2): LightGBM qua B0 (code gốc, không sửa), XGBoost, XGB-RF, CatBoost.

Giao diện chung: fit_predict(X_fit, z_fit, X_es, z_es, X_pred, rounds, seed) → FitResult.
- z = target sau TargetTransform của B0 (z-space); 3 model độc lập theo horizon, seed + 101·col như B0.
- rounds = None → early stopping trên ES set (run calibrate); rounds = (r1, r2, r3) → số vòng cố định (§1.3).
- Training chỉ GPU (§0): device mặc định GPU; `allow_cpu=True` CHỈ cho unit test.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .config import HORIZONS

# lightgbm ≥ 4.7 cảnh báo `eval_set` deprecated; B0 (frozen) dùng eval_set nên harness giữ nguyên API này cho đồng nhất
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", message=".*np.ndarray subset.*")

Rounds = Sequence[int] | None


@dataclass
class FitResult:
    pred_z: np.ndarray  # (n_pred, 3)
    best_iters: tuple[int, int, int]
    predictors: list[Callable[[np.ndarray], np.ndarray]] = field(default_factory=list)  # per horizon: X → z

    def predict_z(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([p(X) for p in self.predictors]).astype(np.float32)


class TabularModel:
    name: str = "base"
    supports_rounds: bool = True
    train_device: str = "GPU"
    predict_device: str = "CPU"  # device thực tế khi predict (§7.4): LightGBM/CatBoost luôn CPU (đặc tính thư viện)
    lib: str = ""

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds: Rounds, seed: int) -> FitResult:  # pragma: no cover
        raise NotImplementedError


def _cpu_guard(device_is_gpu: bool, allow_cpu: bool, name: str) -> None:
    if not device_is_gpu and not allow_cpu:
        raise RuntimeError(f"{name}: training trên CPU bị cấm (plan §0). Dùng allow_cpu=True chỉ trong unit test.")


class LGBMModel(TabularModel):
    """LightGBM đúng code gốc B0: LGBMConfig + _make_model + ES monitor huber (Baseline_LGBM.fit_lgbm_baseline)."""

    name = "lgbm"
    lib = "lightgbm"
    predict_device = "CPU"

    def __init__(self, device_type: str = "gpu", allow_cpu: bool = False, **overrides):
        from Baseline_LGBM import LGBMConfig

        _cpu_guard(device_type in ("gpu", "cuda"), allow_cpu, "LightGBM")
        self.config = LGBMConfig(require_p100=False, device_type=device_type, **overrides)
        self.train_device = "GPU" if device_type in ("gpu", "cuda") else "CPU"

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds: Rounds, seed: int) -> FitResult:
        import lightgbm as lgb
        from Baseline_LGBM import _make_model

        preds = np.empty((len(X_pred), len(HORIZONS)), dtype=np.float32)
        best, predictors = [], []
        for col in range(len(HORIZONS)):
            n_est = int(rounds[col]) if rounds is not None else None
            model = _make_model(self.config, seed=seed + 101 * col, n_estimators=n_est)
            kwargs = {"callbacks": [lgb.log_evaluation(period=0)]}
            if n_est is None:
                kwargs["eval_set"] = [(X_es, z_es[:, col])]
                kwargs["eval_metric"] = "huber"
                kwargs["callbacks"] = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)]
            model.fit(X_fit, z_fit[:, col], **kwargs)
            bi = int(getattr(model, "best_iteration_", 0) or n_est or self.config.n_estimators)
            preds[:, col] = model.predict(X_pred, num_iteration=bi)
            best.append(bi)
            predictors.append(lambda X, m=model, b=bi: np.asarray(m.predict(X, num_iteration=b), dtype=np.float32))
        return FitResult(preds, tuple(best), predictors)


class XGBModel(TabularModel):
    """XGBoost: hist, device cuda, reg:pseudohubererror (huber_slope 0.9), lr 0.03, max_depth 6, ES 80 (§2.2 #2)."""

    name = "xgb"
    lib = "xgboost"

    def __init__(self, device: str = "cuda", allow_cpu: bool = False, learning_rate: float = 0.03, max_depth: int = 6,
                 n_estimators: int = 1200, early_stopping_rounds: int = 80, huber_slope: float = 0.9):
        _cpu_guard(device == "cuda", allow_cpu, "XGBoost")
        self.device, self.lr, self.max_depth = device, learning_rate, max_depth
        self.n_estimators, self.es_rounds, self.huber_slope = n_estimators, early_stopping_rounds, huber_slope
        self.train_device = self.predict_device = "GPU" if device == "cuda" else "CPU"

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds: Rounds, seed: int) -> FitResult:
        import xgboost as xgb

        preds = np.empty((len(X_pred), len(HORIZONS)), dtype=np.float32)
        best, predictors = [], []
        for col in range(len(HORIZONS)):
            n_est = int(rounds[col]) if rounds is not None else self.n_estimators
            model = xgb.XGBRegressor(
                objective="reg:pseudohubererror", huber_slope=self.huber_slope, learning_rate=self.lr, max_depth=self.max_depth,
                n_estimators=n_est, subsample=0.85, colsample_bytree=0.85, reg_alpha=1e-4, reg_lambda=1e-2,
                tree_method="hist", device=self.device, random_state=seed + 101 * col,
                early_stopping_rounds=(self.es_rounds if rounds is None else None), verbosity=0,
            )
            if rounds is None:
                model.fit(X_fit, z_fit[:, col], eval_set=[(X_es, z_es[:, col])], verbose=False)
                bi = int(model.best_iteration) + 1
            else:
                model.fit(X_fit, z_fit[:, col], verbose=False)
                bi = n_est
            preds[:, col] = model.predict(X_pred, iteration_range=(0, bi))
            best.append(bi)
            predictors.append(lambda X, m=model, b=bi: np.asarray(m.predict(X, iteration_range=(0, b)), dtype=np.float32))
        return FitResult(preds, tuple(best), predictors)


class XGBRFModel(TabularModel):
    """XGB-RF (thay ExtraTrees, §2.2 #5): random-forest mode trên GPU, 1 vòng boosting, không ES, không calibrate số vòng."""

    name = "xgbrf"
    lib = "xgboost"
    supports_rounds = False

    def __init__(self, device: str = "cuda", allow_cpu: bool = False, n_estimators: int = 500, max_depth: int = 8,
                 min_child_weight: float = 500.0, subsample: float = 0.63, colsample_bynode: float = 0.3):
        _cpu_guard(device == "cuda", allow_cpu, "XGB-RF")
        self.device = device
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight,
                           subsample=subsample, colsample_bynode=colsample_bynode)
        self.train_device = self.predict_device = "GPU" if device == "cuda" else "CPU"

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds: Rounds, seed: int) -> FitResult:
        import xgboost as xgb

        preds = np.empty((len(X_pred), len(HORIZONS)), dtype=np.float32)
        predictors = []
        for col in range(len(HORIZONS)):
            model = xgb.XGBRFRegressor(objective="reg:squarederror", learning_rate=1.0, tree_method="hist", device=self.device,
                                       random_state=seed + 101 * col, verbosity=0, **self.params)
            model.fit(X_fit, z_fit[:, col])
            preds[:, col] = model.predict(X_pred)
            predictors.append(lambda X, m=model: np.asarray(m.predict(X), dtype=np.float32))
        return FitResult(preds, (1, 1, 1), predictors)


class CatBoostModel(TabularModel):
    """CatBoost: GPU, Huber:delta=0.9, lr 0.03, depth 6, ES 80 (§2.2 #3). Nếu GPU không hỗ trợ Huber → researcher audit."""

    name = "cat"
    lib = "catboost"
    predict_device = "CPU"

    def __init__(self, task_type: str = "GPU", allow_cpu: bool = False, learning_rate: float = 0.03, depth: int = 6,
                 iterations: int = 1200, early_stopping_rounds: int = 80, loss_function: str = "Huber:delta=0.9"):
        _cpu_guard(task_type == "GPU", allow_cpu, "CatBoost")
        self.task_type, self.lr, self.depth = task_type, learning_rate, depth
        self.iterations, self.es_rounds, self.loss = iterations, early_stopping_rounds, loss_function
        self.train_device = "GPU" if task_type == "GPU" else "CPU"

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds: Rounds, seed: int) -> FitResult:
        from catboost import CatBoostRegressor

        preds = np.empty((len(X_pred), len(HORIZONS)), dtype=np.float32)
        best, predictors = [], []
        for col in range(len(HORIZONS)):
            n_it = int(rounds[col]) if rounds is not None else self.iterations
            kwargs = dict(loss_function=self.loss, learning_rate=self.lr, depth=self.depth, iterations=n_it,
                          task_type=self.task_type, random_seed=seed + 101 * col, verbose=False, allow_writing_files=False)
            if rounds is None:
                kwargs.update(od_type="Iter", od_wait=self.es_rounds)
            model = CatBoostRegressor(**kwargs)
            if rounds is None:
                model.fit(X_fit, z_fit[:, col], eval_set=(X_es, z_es[:, col]), use_best_model=True)
                bi = int(model.get_best_iteration() or 0) + 1
            else:
                model.fit(X_fit, z_fit[:, col])
                bi = n_it
            preds[:, col] = model.predict(X_pred)
            best.append(bi)
            predictors.append(lambda X, m=model: np.asarray(m.predict(X), dtype=np.float32))
        return FitResult(preds, tuple(best), predictors)


def make_model(name: str, params: dict | None = None, allow_cpu: bool = False) -> TabularModel:
    params = dict(params or {})
    if name == "lgbm":
        return LGBMModel(allow_cpu=allow_cpu, **params)
    if name == "xgb":
        return XGBModel(allow_cpu=allow_cpu, **params)
    if name == "xgbrf":
        return XGBRFModel(allow_cpu=allow_cpu, **params)
    if name == "cat":
        return CatBoostModel(allow_cpu=allow_cpu, **params)
    if name == "lstm":
        from .models_lstm import LSTMModel

        return LSTMModel(allow_cpu=allow_cpu, **params)
    if name in ("tfm", "autots_wr", "autots_mr"):
        from .models_pending import pending

        return pending(name)
    raise KeyError(f"model không có trong plan §2.2: {name}")
