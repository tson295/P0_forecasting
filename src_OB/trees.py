"""Existing four tree families; one separately fitted model per horizon."""
from __future__ import annotations

import numpy as np

from .gpu import GPURegressor


def matrix(data, ids):
    # Materialize one fixed feature matrix per fold, reused by all tree families/horizons.
    return np.concatenate([data.flat(ids[s:s + 4096]) for s in range(0, len(ids), 4096)])


def make_model(name, cfg, seed):
    p = cfg["tree"]
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(device_type=p["lightgbm_device"], objective="huber", alpha=0.9,
                             n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                             max_depth=p["max_depth"], num_leaves=31, random_state=seed, verbosity=-1)
    if name in ("xgb", "xgbrf"):
        from xgboost import XGBRegressor
        if name == "xgbrf":
            return XGBRegressor(device="cuda", tree_method="hist", objective="reg:squarederror",
                                n_estimators=1, num_parallel_tree=500, learning_rate=1,
                                max_depth=8, subsample=0.63, colsample_bynode=0.3, random_state=seed)
        return XGBRegressor(device="cuda", tree_method="hist", objective="reg:pseudohubererror",
                            huber_slope=0.9, n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                            learning_rate=p["learning_rate"], random_state=seed)
    if name == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(task_type="GPU", devices="0", loss_function="Huber:delta=0.9",
                                 iterations=p["n_estimators"], learning_rate=p["learning_rate"],
                                 depth=p["max_depth"], random_seed=seed, verbose=False,
                                 allow_writing_files=False)
    raise KeyError(name)


def run(name, cfg, x_train, y_train, x_val, out):
    import joblib
    # Constant train-only scaling; no global volatility from unseen future data.
    scale = max(float(np.std(y_train)), 1e-8)
    model = GPURegressor(make_model(name, cfg, cfg["seed"]), name)
    model.fit(x_train, y_train / scale)
    pred = np.asarray(model.predict(x_val), np.float64) * scale
    joblib.dump({"model": model, "target_scale": scale}, out / "model.joblib")
    return pred
