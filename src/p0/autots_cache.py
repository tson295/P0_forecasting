"""Immutable CPU preprocessing for AutoTS feature search, built before any fit.

Cache the complete feature pool per fold. Candidate fits only select columns and
the seed's native WR bootstrap rows. MR redundancy decisions remain conditional
on the selected, ordered columns; only their expensive statistics are cached.
TimesFM reuses the fold covariate pools only; its adapter/forecast cache is separate.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HORIZONS
from .models import SeriesBatch
from .models_tfm_residual import atomic_json, digest, grid_digest

SCHEMA = "autots-cpu-pool-v1"
_OPEN: OrderedDict[str, object] = OrderedDict()
ROLLING_PARAMS = (
    "mean_rolling_periods", "macd_periods", "std_rolling_periods", "max_rolling_periods",
    "min_rolling_periods", "ewm_var_alpha", "quantile90_rolling_periods", "quantile10_rolling_periods",
    "additional_lag_periods", "ewm_alpha", "abs_energy", "rolling_autocorr_periods", "nonzero_last_n",
    "window", "rolling_skew_periods", "diff_periods", "rolling_range_periods")


def _code_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for source in sorted(root.glob("*.py")) + [root.parent.parent / "Baseline_LGBM.py"]:
        h.update(source.name.encode())
        h.update(source.read_bytes())
    return h.hexdigest()


def _data_hash(store):
    if not hasattr(store, "_autots_data_hash"):
        seq = SeriesBatch(store.ts, store.r1, np.array([], dtype=np.int64))
        h = hashlib.sha256((grid_digest(seq) + digest(store.eligible)).encode())
        # Volume/OHLC/5-minute inputs can change while r1 stays identical.
        for frame in (store.grid, store.fd.frame, store.raw_lf):
            if frame is not None:
                h.update(repr(tuple(frame.columns)).encode())
                h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        store._autots_data_hash = h.hexdigest()
    return store._autots_data_hash


def _dependencies():
    from importlib.metadata import version

    return {name: version(name) for name in ("autots", "numpy", "pandas")}


def _frame(ts, r1):
    return pd.DataFrame({"r1": r1}, index=pd.to_datetime(ts, unit="s"))


def recipe(model):
    if model.lib == "timesfm":
        return None, {"kind": "tfm_covariates", "standardization": "fit-only-float32", "schema": SCHEMA}
    from importlib.metadata import version

    if version("autots") != "1.0.4":
        raise ValueError("AutoTS cache requires autots==1.0.4")
    m = model._make(0)  # construction only: no fit, probe, or model execution
    m.column_names = pd.Index(["r1"])
    from .autots_batch import _guard

    _guard(m, model.kind)
    if model.kind == "wr":
        if m.input_dim != "univariate" or m.scale or m.normalize_window or m.fourier_encoding_components is not None:
            raise ValueError("WR preprocessing cache requires unscaled univariate windows without Fourier encoding")
    elif (m.scale_full_X or m.frac_slice is not None or m.discard_data is not None
          or m.synthetic_boundary_ratio or m.series_hash):
        raise ValueError("MR preprocessing cache does not accept scaling/slicing/discard/synthetic/hash options")
    params = m.get_params().copy()
    params.pop("regression_model", None)  # estimator/seed do not change feature definitions
    return m, {"kind": model.kind, "params": params, "tail_bars": model.tail_bars,
               "forecast_length": len(HORIZONS), "schema": SCHEMA}


def _save(path, name, array):
    np.save(path / f"{name}.npy", np.asarray(array), allow_pickle=False)


def _mr_statistics(x):
    lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
    valid = ((hi - lo) > 1e-10) & np.isfinite(hi - lo)
    groups, lookup = np.empty(x.shape[1], np.int64), {}
    # Native MR: DataFrame(X_subset).T.round(10).duplicated(). Compute equality
    # groups once, not a global keep-mask (a candidate may exclude an earlier twin).
    for col in range(x.shape[1]):
        a = np.round(np.array(x[:, col], dtype=np.float64), 10)
        a[a == 0] = 0.0
        a[np.isnan(a)] = np.nan
        key = hashlib.sha256(a.tobytes()).hexdigest()
        if key not in lookup:
            lookup[key] = len(lookup)
        groups[col] = lookup[key]
    # Pairwise statistics are FIT-only. Apply the native 2,000-column condition
    # to each selected candidate below, not to the full feature pool.
    corr = np.abs(np.corrcoef(x, rowvar=False))
    return valid, groups, np.atleast_2d(corr)


def prepare_cache(cfg, store, folds, model, base, candidates, log=None):
    """Build all folds/seed selectors, then publish READY. Never fit an estimator."""
    from .harness import ColSet
    from autots.models.sklearn import rolling_x_regressor
    from autots.tools.window_functions import window_maker

    root = Path(model.preprocess_cache_dir)
    native, spec = recipe(model)
    if not folds:
        raise ValueError("Cannot prepare AutoTS cache without folds")
    ext = tuple(dict.fromkeys((*base.ext, *(col for c in candidates for col in c.columns))))
    pool = ColSet(base.b0, ext)
    seeds = sorted({int(cfg.calib_seed), int(cfg.sel_seed), *map(int, cfg.eval_seeds)})
    indices = {f.name: tuple(p.origins(store.ts, store.eligible) for p in (f.fit, f.es, f.val)) for f in folds}
    contract = {"schema": SCHEMA, "code": _code_hash(), "dependencies": _dependencies(),
                "data": _data_hash(store), "recipe": spec,
                "columns": list(pool.names), "seeds": seeds,
                "fold_indices": {name: digest(*parts) for name, parts in indices.items()}}
    # JSON round trip also normalizes native tuple-valued parameters.
    contract = json.loads(json.dumps(contract, sort_keys=True))
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    generation = root / fingerprint
    ready = generation / "READY.json"
    if ready.exists():
        if json.loads(ready.read_text())["contract"] != contract:
            raise ValueError("AutoTS preprocessing contract mismatch")
        atomic_json(root / "CURRENT.json", {"generation": fingerprint})
        return {"cache_hit": True, "path": str(generation)}
    root.mkdir(parents=True, exist_ok=True)
    work = root / (fingerprint + ".building." + uuid.uuid4().hex)
    work.mkdir()
    started = time.perf_counter()
    # One full-grid extraction for the entire pool, not one per candidate.
    raw = store.grid_matrix(pool)
    summaries = {}
    for fold in folds:
        fold_started = time.perf_counter()
        if log is not None:
            log(f"[{model.name}] CPU cache: {fold.name}, {len(pool.names)} covariate columns")
        fit_idx, es_idx, val_idx = indices[fold.name]
        if any(len(a) == 0 for a in (fit_idx, es_idx, val_idx)):
            raise ValueError(f"Empty AutoTS fold: {fold.name}")
        dest = work / fold.name
        dest.mkdir()
        lo, hi = int(fit_idx[0]), int(fit_idx[-1]) + 1
        if not np.all(np.diff(store.ts[lo:hi]) == 60):
            raise ValueError(f"AutoTS FIT crosses a minute gap: {fold.name}")
        span_end = int(max(es_idx[-1], val_idx[-1], fit_idx[-1])) + 1
        train_raw = raw[fit_idx]
        mean, scale = np.nanmean(train_raw, axis=0), np.nanstd(train_raw, axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        del train_raw
        cov = np.nan_to_num((raw[lo:span_end] - mean) / scale, nan=0, posinf=0, neginf=0).astype(np.float32)
        _save(dest, "cov", cov)
        _save(dest, "mean", mean)
        _save(dest, "scale", scale)
        for name, arr in zip(("fit_idx", "es_idx", "val_idx"), (fit_idx, es_idx, val_idx)):
            _save(dest, name, arr)
        if native is None:  # TimesFM: no AutoTS training matrices or recursive features
            atomic_json(dest / "metadata.json", {
                "lo": lo, "hi": hi, "span_end": span_end, "columns": list(pool.names),
                "base_columns": 0, "recipe": spec, "fold": fold.name,
                "data": contract["data"], "indices_hash": contract["fold_indices"][fold.name]})
            summaries[fold.name] = {
                "fit_rows": len(fit_idx), "val_origins": len(val_idx),
                "array_bytes": sum(p.stat().st_size for p in dest.glob("*.npy")),
                "seconds": time.perf_counter() - fold_started}
            if log is not None:
                log(f"[{model.name}] CPU cache: {fold.name} prepared in {summaries[fold.name]['seconds']:.1f}s")
            continue
        df = _frame(store.ts[lo:hi], store.r1[lo:hi])
        if model.kind == "wr":
            if native.window_size > len(df) - len(HORIZONS) - 1:
                raise ValueError("FIT too short for configured WR window; no implicit window reduction")
            windows, y = window_maker(df, window_size=native.window_size, input_dim="univariate",
                                      output_dim="forecast_length", forecast_length=len(HORIZONS),
                                      max_windows=None, normalize_window=False, shuffle=False)
            count = len(windows)
            origins = np.arange(count) + native.window_size - 1
            x = np.column_stack((windows, cov[origins]))
            width = int(native.window_size)
            for seed in seeds:
                cap = native.max_windows
                if cap is None:
                    rows = np.arange(count)
                else:
                    size = (count if cap == 1 else int(count * cap) if 0 < cap < 1 else min(int(cap), count))
                    # AutoTS chunk_reshape samples with replacement even when cap >= count.
                    rows = np.random.default_rng(seed).choice(count, size=size, replace=True)
                _save(dest, f"rows_{seed}", rows)
            base_columns = width
        else:
            params = {name: getattr(native, name) for name in ROLLING_PARAMS}
            features = rolling_x_regressor(df.iloc[:-1].astype(float), **params)
            base_columns = features.shape[1]
            x = np.column_stack((features.to_numpy(), cov[:len(df) - 1]))
            x = np.clip(x, -1e12, 1e12)
            x[np.isinf(x)] = np.nan
            valid, groups, corr = _mr_statistics(x)
            for name, arr in (("valid", valid), ("groups", groups), ("corr", corr)):
                _save(dest, name, arr)
            x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
            y = np.nan_to_num(np.clip(store.r1[lo + 1:hi], -1e12, 1e12), nan=0, posinf=0, neginf=0)
            width = min(model.tail_bars, int(native.min_threshold))
        _save(dest, "train_x", x)
        _save(dest, "train_y", y)
        del x, y
        positions = val_idx[:, None] - np.arange(width - 1, -1, -1)
        if positions.min() < 0 or not np.all(store.ts[positions] == store.ts[val_idx, None] - np.arange(width - 1, -1, -1) * 60):
            raise ValueError("Cached prediction context crosses a minute gap")
        histories = np.asarray(store.r1[positions], dtype=np.float64)
        _save(dest, "histories", histories)
        if model.kind == "mr":
            blocks = []
            for start in range(0, len(val_idx), model.predict_batch_size):
                h = histories[start:start + model.predict_batch_size].T
                index = pd.to_datetime(store.ts[positions[start]], unit="s")
                result = rolling_x_regressor(pd.DataFrame(h, index=index), **params)
                blocks.append(result.iloc[-1].to_numpy().reshape(-1, h.shape[1]).T)
            _save(dest, "first_step", np.concatenate(blocks))
        meta = {"lo": lo, "hi": hi, "span_end": span_end, "columns": list(pool.names),
                "base_columns": base_columns, "recipe": spec, "fold": fold.name,
                "fit_rows": int(len(df)), "val_origins": len(val_idx), "data": contract["data"],
                "indices_hash": contract["fold_indices"][fold.name]}
        atomic_json(dest / "metadata.json", meta)
        summaries[fold.name] = {"fit_rows": len(df), "val_origins": len(val_idx), "base_columns": base_columns,
                                "array_bytes": sum(p.stat().st_size for p in dest.glob("*.npy")),
                                "seconds": time.perf_counter() - fold_started}
        if log is not None:
            log(f"[{model.name}] CPU cache: {fold.name} prepared in {summaries[fold.name]['seconds']:.1f}s")
    del raw
    report = {"contract": contract, "folds": summaries, "preprocessing_seconds": time.perf_counter() - started}
    atomic_json(work / "READY.json", report)
    os.rename(work, generation)  # .building remains evidence if preprocessing fails
    atomic_json(root / "CURRENT.json", {"generation": fingerprint})
    return {"cache_hit": False, "path": str(generation), **report}


class ColumnView:
    """Select rows/columns from a compact memmap using global timeline indices."""
    def __init__(self, block, positions, grid_size):
        self.block, self.positions = block, np.asarray(positions, np.int64)
        self.shape = (grid_size, len(positions))

    def __getitem__(self, key):
        rows, cols = key if isinstance(key, tuple) else (key, slice(None))
        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(self.shape[0]))
        local = np.asarray(rows) - self.block.meta["lo"]
        if np.any(local < 0) or np.any(local >= len(self.block.cov)):
            raise ValueError("Covariate access outside cached fold span")
        selected = self.positions[cols]
        if np.ndim(selected) == 0:
            return self.block.cov[local, selected]
        return self.block.cov[local[..., None], selected]


class PreparedFold:
    def __init__(self, path):
        self.path = path
        self.meta = json.loads((path / "metadata.json").read_text())
        self.arrays = {}
        self.cov = self.array("cov")
        self.positions = {name: i for i, name in enumerate(self.meta["columns"])}
        self.checked_partitions = set()

    def array(self, name):
        if name not in self.arrays:
            self.arrays[name] = np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        return self.arrays[name]

    def prediction_rows(self, idx):
        original = self.array("val_idx")
        rows = np.searchsorted(original, idx)
        if np.any(rows >= len(original)) or not np.array_equal(original[rows], idx):
            raise ValueError("Origins not present in the prepared validation cache")
        return rows

    def design(self, names, seed):
        base = self.meta["base_columns"]
        columns = np.array([*range(base), *(base + self.positions[n] for n in names)], dtype=np.int64)
        native_mask = None
        if self.meta["recipe"]["kind"] == "wr":
            rows = self.array(f"rows_{seed}")
            x = np.asarray(self.array("train_x")[np.ix_(rows, columns)])
            y = np.asarray(self.array("train_y")[rows])
        else:
            valid = np.asarray(self.array("valid")[columns]).copy()
            eligible = np.flatnonzero(valid)
            if not len(eligible):
                raise ValueError("MR candidate has no nonconstant features")
            if 1 < len(columns) < 10000:
                seen = set()
                duplicate = []
                for group in self.array("groups")[columns[eligible]]:
                    duplicate.append(int(group) in seen)
                    seen.add(int(group))
                duplicate = np.array(duplicate)
                if 1 < len(eligible) < 2000:
                    corr = self.array("corr")[np.ix_(columns[eligible], columns[eligible])]
                    duplicate |= np.any(np.triu(corr, k=1) > 0.9999, axis=0)
                valid[eligible[duplicate]] = False
            native_mask = None if valid.all() else valid
            x = np.asarray(self.array("train_x")[:, columns[valid]])
            y = np.asarray(self.array("train_y"))
        return x, y, native_mask


def sequences(model, store, fold, names):
    root = Path(model.preprocess_cache_dir)
    current = json.loads((root / "CURRENT.json").read_text())
    generation = root / current["generation"]
    key = str(generation / fold.name)
    if key not in _OPEN:
        ready = json.loads((generation / "READY.json").read_text())
        contract = ready["contract"]
        if contract["schema"] != SCHEMA or contract["data"] != _data_hash(store):
            raise ValueError("AutoTS cache belongs to another dataset/schema")
        if contract["code"] != _code_hash() or contract.get("dependencies") != _dependencies():
            raise ValueError("AutoTS cache code/dependency versions changed; prepare a new generation")
        _OPEN[key] = PreparedFold(generation / fold.name)
        while len(_OPEN) > 12:
            _OPEN.popitem(last=False)
    block = _OPEN[key]
    _OPEN.move_to_end(key)
    if block.meta["data"] != _data_hash(store):
        raise ValueError("AutoTS cache input data changed")
    if not hasattr(model, "_prepared_recipe"):
        model._prepared_recipe = json.loads(json.dumps(recipe(model)[1], sort_keys=True))
    if block.meta["recipe"] != model._prepared_recipe:
        raise ValueError("AutoTS cache preprocessing parameters changed")
    partition = repr(fold)
    if partition not in block.checked_partitions:
        parts = tuple(p.origins(store.ts, store.eligible) for p in (fold.fit, fold.es, fold.val))
        if digest(*parts) != block.meta["indices_hash"]:
            raise ValueError("AutoTS cache FIT/ES/VAL masks changed")
        block.checked_partitions.add(partition)
    positions = [block.positions[name] for name in names]  # missing pool column is an error, never recomputed here
    cov = ColumnView(block, positions, len(store.ts))
    return tuple(SeriesBatch(store.ts, store.r1, block.array(label), cov, tuple(names), prepared=block)
                 for label in ("fit_idx", "es_idx", "val_idx"))


def fit_prepared(wrapper, native, seq, seed):
    """Only column/row selection plus a real GPU estimator fit, no native preprocessing fit."""
    from .autots_batch import AutoTSGPURegressor
    from autots.models.sklearn import retrieve_regressor
    import torch

    block = seq.prepared
    started = time.perf_counter()
    x, y, mask = block.design(seq.cov_names, seed)
    selection_seconds = time.perf_counter() - started
    native.column_names = pd.Index(["r1"])
    native.train_shape = (block.meta["hi"] - block.meta["lo"], 1)
    native.train_last_date = pd.Timestamp(int(seq.ts[block.meta["hi"] - 1]), unit="s")
    if wrapper.kind == "mr":
        native._nonzero_var_mask = mask
        native.multioutputgpr = False
    spec = native.regression_model
    if spec["model"] not in ("LightGBM", "xgboost"):
        raise ValueError("Cached AutoTS fit requires a GPU allowlisted estimator")
    estimator = retrieve_regressor(regression_model=spec, verbose=native.verbose,
                                    verbose_bool=native.verbose_bool, random_seed=seed,
                                    n_jobs=native.n_jobs, multioutput=y.ndim == 2 and y.shape[1] > 1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    native.model = AutoTSGPURegressor(estimator, "lgbm" if spec["model"] == "LightGBM" else "xgb").fit(x, y)
    torch.cuda.synchronize()
    return {"selection_seconds": selection_seconds, "gpu_fit_seconds": time.perf_counter() - started,
            "train_rows": x.shape[0], "train_columns": x.shape[1]}
