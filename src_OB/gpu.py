"""Device policy during real fits, with no trial fits or CPU retry path."""
from __future__ import annotations

import json
import warnings

import numpy as np


class GPUOnlyError(SystemExit):
    """Stop the job: AutoTS must not swallow device failures as bad candidates."""


class GPURegressor:
    """CUDA input for XGBoost; explicit GPU backends for LightGBM/CatBoost."""

    def __init__(self, estimator, family):
        self.estimator = estimator
        self.family = family

    def _array(self, value):
        if self.family in ("xgb", "xgbrf"):
            import cupy as cp
            return cp.asarray(np.asarray(value, dtype=np.float32))
        return value

    def fit(self, x, y, **kwargs):
        import torch
        if not torch.cuda.is_available():
            raise GPUOnlyError("CUDA unavailable; CPU training is forbidden.")
        params = self.estimator.get_params()
        if self.family in ("xgb", "xgbrf"):
            import xgboost
            if not xgboost.build_info().get("USE_CUDA", False):
                raise GPUOnlyError("XGBoost was built without CUDA.")
            if params.get("device") != "cuda" or params.get("tree_method") != "hist":
                raise GPUOnlyError("XGBoost must use device=cuda, tree_method=hist.")
        elif self.family == "lgbm":
            if params.get("device_type") not in ("gpu", "cuda"):
                raise GPUOnlyError("LightGBM must use a GPU backend.")
        elif self.family == "cat":
            if params.get("task_type") != "GPU":
                raise GPUOnlyError("CatBoost must use task_type=GPU.")
        else:
            raise GPUOnlyError(f"Unapproved estimator: {self.family}")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="(?i).*(no visible gpu|not compiled with cuda|falling back|setting device to cpu).*")
                self.estimator.fit(self._array(x), self._array(y), **kwargs)
        except Exception as exc:
            if any(word in str(exc).lower() for word in ("cuda", "gpu", "opencl", "falling back")):
                raise GPUOnlyError(f"GPU fit failed; no CPU retry: {exc}") from exc
            raise
        if self.family in ("xgb", "xgbrf"):
            state = json.loads(self.estimator.get_booster().save_config())
            if not state["learner"]["generic_param"]["device"].startswith("cuda"):
                raise GPUOnlyError("XGBoost did not retain its required CUDA device.")
        return self

    def predict(self, x, **kwargs):
        result = self.estimator.predict(self._array(x), **kwargs)
        if hasattr(result, "get"):
            result = result.get()
        return np.asarray(result)
