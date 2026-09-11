"""Device policy during real fits, with no trial fits or CPU retry path."""
from __future__ import annotations

import json
import re
import warnings

import numpy as np

# LightGBM 4.7.0 CUDA limits (src/io/config.cpp, objective/objective_function.cpp, boosting/goss.hpp): linear trees
# silently switch the whole fit to CPU, these objectives compute gradients on CPU, GOSS reads an unallocated CUDA buffer.
LGBM_CPU_OBJECTIVES = {"cross_entropy", "cross_entropy_lambda", "mape", "gamma", "tweedie"}


def lightgbm_effective_config(estimator):
    """Config LightGBM actually trained with, after its own conflict checks (parameters block of the model text)."""
    text = estimator.booster_.model_to_string(num_iteration=1)
    section = text.split("\nparameters:\n", 1)[1].split("\nend of parameters", 1)[0]
    return dict(re.findall(r"^\[([^:\]]+): ?(.*)\]$", section, re.M))


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
            if (params.get("linear_tree") or "goss" in (params.get("boosting_type"), params.get("data_sample_strategy"))
                    or params.get("objective") in LGBM_CPU_OBJECTIVES):
                raise GPUOnlyError("LightGBM option outside its CUDA backend (linear_tree/GOSS/CPU objective).")
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
        elif self.family == "lgbm":
            used = lightgbm_effective_config(self.estimator)
            if (used.get("device_type") not in ("gpu", "cuda") or used.get("linear_tree") != "0"
                    or used.get("data_sample_strategy") == "goss" or used.get("objective") in LGBM_CPU_OBJECTIVES):
                raise GPUOnlyError("LightGBM did not train on its CUDA backend: " + str(
                    {k: used.get(k) for k in ("device_type", "linear_tree", "data_sample_strategy", "objective")}))
        return self

    def predict(self, x, **kwargs):
        result = self.estimator.predict(self._array(x), **kwargs)
        if hasattr(result, "get"):
            result = result.get()
        return np.asarray(result)
