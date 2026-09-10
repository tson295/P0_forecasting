"""Native AutoTS search over GPU regressors using fixed observed OF features.

The scoped adapter aligns features with each candidate's window and enforces
devices. AutoTS generates, scores and selects model parameters. No CPU fitting,
feature subset search, recursive horizon or outer-VAL tuning.
"""
from __future__ import annotations

import copy
import json
import random
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .config import write_json
from .data import DAY
from .gpu import GPUOnlyError, GPURegressor


def frame(ts, values):
    return pd.DataFrame(values, index=pd.to_datetime(ts, unit="us"), columns=["mid_price"])


def gpu_parameters(spec, cfg):
    spec = copy.deepcopy(spec)
    name, p = spec["model"], spec["model_params"]
    for key in ("n_jobs", "random_state", "verbose", "verbosity", "device", "device_type"):
        p.pop(key, None)
    if name == "LightGBM":
        p.update(device_type=cfg["tree"]["lightgbm_device"], objective="regression")
    elif name == "xgboost":
        for key in ("gpu_id", "predictor", "updater", "quantile_alpha", "multi_strategy"):
            p.pop(key, None)
        p.update(device="cuda", tree_method="hist", booster="gbtree", objective="reg:squarederror")
    else:
        raise GPUOnlyError(f"AutoTS candidate {name} is outside the GPU allowlist.")
    return spec


@contextmanager
def gpu_search_space(cfg, data, horizon):
    import autots.models.sklearn as native
    import autots.evaluator.auto_model as factory

    original_class = native.WindowRegression
    original_factory_class = factory.WindowRegression
    original_retrieve = native.retrieve_regressor
    step = horizon * 1_000_000

    class ObservedOrderBookRegression(original_class):
        def get_new_params(self, method="random"):
            params = super().get_new_params(method="regressor")
            spec = native.generate_regressor_params(
                model_dict={"LightGBM": 0.5, "xgboost": 0.5}, method="random")
            params.update(regression_model=gpu_parameters(spec, cfg),
                          window_size=min(params["window_size"], cfg["autots"]["max_window_size"]),
                          input_dim="univariate", output_dim="forecast_length", regression_type="User",
                          datepart_method=None, scale=False, normalize_window=False,
                          fourier_encoding_components=None, max_windows=cfg["autots"]["max_windows"])
            return params

        def fit(self, df, future_regressor=None):
            if self.regression_type != "User" or self.scale or self.datepart_method is not None:
                raise GPUOnlyError("AutoTS candidate changed the fixed OF input contract.")
            self.regression_model = gpu_parameters(self.regression_model, cfg)
            # AutoTS evaluates a block of origins; the fitted head remains scalar.
            self.forecast_length = 1
            # R at target s contains features from s-h. Native window_maker
            # reads R at window START. Align within this TRAIN partition only.
            reg = future_regressor.reindex(df.index).shift(-self.window_size).fillna(0.0)
            return super().fit(df, future_regressor=reg)

        def predict(self, forecast_length=None, future_regressor=None, just_point_forecast=False, df=None):
            if df is not None:
                return super().predict(forecast_length=1, future_regressor=future_regressor,
                                       just_point_forecast=just_point_forecast, df=df)
            points = []
            for row in range(len(future_regressor)):
                reg = future_regressor.iloc[row:row + 1]
                # Each row is a separate observed origin: never feed predictions
                # back as context or read prices/features at target t+h.
                target = reg.index[0].value // 1000
                times = target - np.arange(self.window_size, 0, -1) * step
                prices, valid, _ = data.prices_at(times)
                if not valid.all():
                    raise ValueError("AutoTS validation context has missing historical quotes.")
                points.append(super().predict(forecast_length=1, future_regressor=reg,
                              just_point_forecast=just_point_forecast, df=frame(times, prices)))
            if just_point_forecast:
                return pd.concat(points)
            result = points[0]
            runtime = sum((p.predict_runtime for p in points), result.predict_runtime * 0)
            for key in ("forecast", "lower_forecast", "upper_forecast"):
                setattr(result, key, pd.concat([getattr(p, key) for p in points]))
            result.forecast_length = len(points)
            result.forecast_index = result.forecast.index
            result.predict_runtime = runtime
            return result

    def retrieve(regression_model, *args, **kwargs):
        spec = gpu_parameters(regression_model, cfg)
        if kwargs.get("multioutput", False):
            raise GPUOnlyError("AutoTS must produce one direct scalar horizon.")
        estimator = original_retrieve(spec, *args, **kwargs)
        return GPURegressor(estimator, "lgbm" if spec["model"] == "LightGBM" else "xgb")

    # Scoped integration for pinned AutoTS. No edits to library files or src/p0.
    native.WindowRegression = factory.WindowRegression = ObservedOrderBookRegression
    native.retrieve_regressor = retrieve
    try:
        yield ObservedOrderBookRegression
    finally:
        native.WindowRegression = original_class
        factory.WindowRegression = original_factory_class
        native.retrieve_regressor = original_retrieve


def run(cfg, data, fold, val_ids, horizon, out):
    import joblib
    from autots import AutoTS

    options = cfg["autots"]
    step = horizon * 1_000_000
    start = ((fold.train_start + step - 1) // step) * step + step
    grid = np.arange(start, fold.train_end, step)
    prices, ok, _ = data.prices_at(grid)
    if not ok.all():
        raise ValueError("AutoTS training grid has missing quotes; supply complete fold data.")
    ids = np.searchsorted(data.ts, grid - step, side="right") - 1
    if (ids < 0).any():
        raise ValueError("AutoTS has no observed order-book covariates.")
    df = frame(grid, prices)  # Native search also scores raw-price RMSE.
    reg = pd.DataFrame(np.asarray(data.features[ids], np.float64), index=df.index)
    validations = []
    for n in range(options["num_validations"] + 1):
        target = grid[-1] - n * cfg["val_days"] * DAY
        target = grid[np.searchsorted(grid, target, side="right") - 1]
        targets = target - np.arange(options["validation_points"] - 1, -1, -1) * step
        train_index = df.index[grid < targets[0] - cfg["gap_days"] * DAY]
        if len(train_index) <= options["max_window_size"] + 1:
            raise ValueError("Not enough FIT history for purged AutoTS search splits.")
        validations.append((train_index, pd.to_datetime(targets, unit="us")))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    with gpu_search_space(cfg, data, horizon) as Regression:
        generator = Regression()
        identity = {"fillna": None, "transformations": {}, "transformation_params": {}}
        templates = pd.DataFrame([
            {"Model": "WindowRegression", "ModelParameters": json.dumps(generator.get_new_params()),
             "TransformationParameters": json.dumps(identity), "Ensemble": 0}
            for _ in range(options["initial_candidates"])
        ])
        search = AutoTS(
            forecast_length=options["validation_points"], frequency=f"{horizon}s", model_list=["WindowRegression"],
            initial_template=templates, max_generations=options["max_generations"],
            num_validations=options["num_validations"], validation_method="custom",
            models_to_validate=0.99, max_per_model_class=100, ensemble=None,
            transformer_list={"None": 1.0}, transformer_max_depth=0,
            metric_weighting={"rmse_weighting": 1}, models_mode="regressor",
            holiday_country=None, random_seed=cfg["seed"], n_jobs=1, verbose=1)
        search.fit(df, future_regressor=reg, validation_indexes=validations)
        search.results().to_csv(out / "search_results.csv", index=False)
        params = copy.deepcopy(search.best_model_params)
        params["regression_model"] = gpu_parameters(params["regression_model"], cfg)
        write_json(out / "selected_model.json", {"family": "WindowRegression", "parameters": params,
                   "search": options, "score": "raw_price_RMSE", "gap_days": cfg["gap_days"],
                   "regressor": "OF/OFI and elapsed at origin only"})
        model = Regression(**params, forecast_length=1, frequency=f"{horizon}s",
                           random_seed=cfg["seed"], n_jobs=1)
        model.fit(df, future_regressor=reg)
        # Save estimator instead of a local class closure containing dataset memmaps.
        joblib.dump({"estimator": model.model, "parameters": params,
                     "feature_names": data.meta["features"], "horizon_seconds": horizon,
                     "input_order": "oldest-to-newest raw mid window, then origin features",
                     "output": "raw mid price"}, out / "model.joblib")
        predicted = np.empty(len(val_ids), dtype=np.float64)
        for j, origin in enumerate(val_ids):
            t = int(data.ts[origin])
            reg_at_origin = pd.DataFrame(np.asarray(data.features[origin:origin + 1], np.float64),
                                        index=pd.to_datetime([t + step], unit="us"))
            point = model.predict(1, future_regressor=reg_at_origin, just_point_forecast=True)
            price = float(np.asarray(point).reshape(-1)[0])
            if not np.isfinite(price) or price <= 0:
                raise ValueError("AutoTS predicted a non-positive or non-finite raw price.")
            predicted[j] = np.log(price / data.mid[origin])
    return predicted
