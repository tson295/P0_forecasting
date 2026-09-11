"""TimesFM first, then a GPU residual regression on held-out historical forecasts.

The last calibration_days of FIT train the residual heads. The preceding
early_stopping_days select LoRA epochs; only the earlier FIT prefix trains LoRA.
No outer ES/VAL labels train either component. Forecasts are independent of the
candidate covariates and are cached by data, adapter and forecasting contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import weakref
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import HORIZONS
from .models import FitResult, SeriesBatch
from .models_tfm import TimesFMLoRAModel, _ADAPTER_META

_GRID_HASHES: dict[tuple[int, int], tuple[object, object, str]] = {}
_FORECASTS: OrderedDict[str, np.ndarray] = OrderedDict()
_CACHE_BYTES = 128 * 1024 * 1024
CONTRACT = "tfm-first-heldout-residual-v1"


def digest(*arrays) -> str:
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def grid_digest(seq: SeriesBatch) -> str:
    key = (id(seq.ts), id(seq.r1))
    hit = _GRID_HASHES.get(key)
    if hit is not None and hit[0]() is seq.ts and hit[1]() is seq.r1:
        return hit[2]
    for old, entry in list(_GRID_HASHES.items()):
        if entry[0]() is None or entry[1]() is None:
            del _GRID_HASHES[old]
    value = digest(seq.ts, seq.r1)
    _GRID_HASHES[key] = (weakref.ref(seq.ts), weakref.ref(seq.r1), value)
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


class TimesFMResidualModel(TimesFMLoRAModel):
    def __init__(self, residual: dict | None = None, preprocess_cache_dir: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.preprocess_cache_dir = preprocess_cache_dir
        self.residual = {"calibration_days": 5, "early_stopping_days": 5,
                         "purge_minutes": 60, "ridge": 0.0, "pinv_rtol": 1e-10,
                         "include_forecast": True, **(residual or {})}
        if self.device != "cuda":
            raise ValueError("TimesFM residual training requires CUDA; no CPU fallback")
        if self.batch_size < 1:
            raise ValueError("TimesFM batch_size must be positive")
        if min(float(self.residual[k]) for k in ("calibration_days", "early_stopping_days")) <= 0:
            raise ValueError("Residual calibration/early-stopping periods must be positive")
        if float(self.residual["purge_minutes"]) < max(HORIZONS):
            raise ValueError("Residual purge must cover the longest label horizon")
        if float(self.residual["ridge"]) < 0 or float(self.residual["pinv_rtol"]) <= 0:
            raise ValueError("Invalid residual solver configuration")

    def _partition(self, seq: SeriesBatch):
        idx = np.asarray(seq.idx, dtype=np.int64)
        if not len(idx) or np.any(np.diff(idx) <= 0):
            raise ValueError("TimesFM residual FIT origins must be ordered and nonempty")
        ts = seq.ts[idx]
        end = int(ts[-1]) + 60
        cal_start = end - int(float(self.residual["calibration_days"]) * 86400)
        purge = int(float(self.residual["purge_minutes"]) * 60)
        es_end = cal_start - purge
        es_start = es_end - int(float(self.residual["early_stopping_days"]) * 86400)
        train_end = es_start - purge
        label_end = ts + max(HORIZONS) * 60
        groups = ((ts < train_end) & (label_end < train_end),
                  (ts >= es_start) & (label_end < es_end),
                  (ts >= cal_start) & (label_end < end))
        parts = tuple(SeriesBatch(seq.ts, seq.r1, idx[mask], seq.cov, seq.cov_names) for mask in groups)
        if any(not len(p.idx) for p in parts):
            raise ValueError("FIT too short for causal LoRA train/early-stop/residual partitions")
        meta = {"contract": CONTRACT, "purge_seconds": purge,
                "partitions": {name: {"first_origin": int(seq.ts[p.idx[0]]),
                                      "last_origin": int(seq.ts[p.idx[-1]]), "n": len(p.idx)}
                               for name, p in zip(("lora_fit", "lora_es", "residual_fit"), parts)}}
        return (*parts, meta)

    def _cache_key(self):
        return super()._cache_key() + json.dumps(self.forecast_config_kwargs(False), sort_keys=True)

    def adapter_key(self, X_fit, X_es, seed, epochs):
        identity = digest(X_fit.idx, X_es.idx if X_es is not None else np.array([], dtype=np.int64))
        contract = json.dumps({"data": grid_digest(X_fit), "origins": identity,
                               "residual": self.residual, "forecast": self.forecast_config_kwargs(False)}, sort_keys=True)
        suffix = hashlib.sha256(contract.encode()).hexdigest()[:20]
        return CONTRACT + "_" + super().adapter_key(X_fit, X_es, seed, epochs) + "_" + suffix

    def _base_prediction(self, adapter: str, seq: SeriesBatch) -> np.ndarray:
        meta = _ADAPTER_META[adapter]
        contract = json.dumps({"adapter": meta["sha256"], "adapter_key": adapter,
                               "repo": self.repo_id, "revision": self.revision, "data": grid_digest(seq),
                               "origins": digest(seq.idx), "forecast": self.forecast_config_kwargs(False),
                               "mean_head": self.use_mean_head, "contract": CONTRACT}, sort_keys=True)
        key = hashlib.sha256(contract.encode()).hexdigest()
        if key in _FORECASTS:
            _FORECASTS.move_to_end(key)
            return _FORECASTS[key]
        path = self.adapter_dir / "forecast_cache" / f"{key}.npy" if self.adapter_dir else None
        if path is not None and path.exists():
            result = np.load(path, allow_pickle=False)
            if result.shape != (len(seq.idx), len(HORIZONS)) or not np.isfinite(result).all():
                raise ValueError(f"Invalid TimesFM forecast cache: {path}")
        else:
            self._load_adapter(adapter)
            m = self._model(False)
            chunks = []
            timings = []
            for start in range(0, len(seq.idx), self.batch_size):
                import torch

                part = seq.slice(start, start + self.batch_size)
                torch.cuda.synchronize()
                started = time.perf_counter()
                chunks.append(np.cumsum(self._point(m, self.contexts(part)), axis=1).astype(np.float32))
                torch.cuda.synchronize()
                timings.append({"start": start, "origins": len(part.idx),
                                "seconds": time.perf_counter() - started})
            result = np.concatenate(chunks) if chunks else np.empty((0, len(HORIZONS)), np.float32)
            self._assert_frozen(adapter)
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite TimesFM prediction")
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
                with tmp.open("wb") as stream:
                    np.save(stream, result, allow_pickle=False)
                os.replace(tmp, path)
                atomic_json(path.with_suffix(".json"), {"contract": json.loads(contract),
                                                         "batches": timings, "cache_hit": False,
                                                         "timing_scope": "real native forecast batches, no warmup"})
        result.setflags(write=False)
        _FORECASTS[key] = result
        while sum(a.nbytes for a in _FORECASTS.values()) > _CACHE_BYTES:
            _FORECASTS.popitem(last=False)
        return result

    def _design(self, seq: SeriesBatch, base: np.ndarray) -> np.ndarray:
        if seq.cov is None:
            x = np.empty((len(seq.idx), 0), dtype=np.float64)
        else:
            x = np.asarray(seq.cov[seq.idx], dtype=np.float64).copy()
            for j, origins in (seq.perm or {}).items():
                x[:, j] = seq.cov[np.asarray(origins, dtype=np.int64), j]
        if self.residual["include_forecast"]:
            x = np.column_stack((x, base))
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite residual features")
        return x

    def fit_predict(self, X_fit, z_fit, X_es, z_es, X_pred, rounds, seed):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for TimesFM and residual regression")
        started = time.perf_counter()
        train, early, calibration, split = self._partition(X_fit)
        epochs = int(rounds[0]) if rounds is not None else None
        key = self.adapter_key(train, early, seed, epochs)
        meta = self._ensure_adapter(key, train, early, seed, epochs)
        # Superclass creates a fresh frozen adapter only on a cache miss.
        self.last_adapter = {"key": key, "sha256": meta["sha256"], "best_epoch": int(meta["best_epoch"]),
                             "mode": meta.get("mode"), "residual_partition": split}
        lora_seconds = time.perf_counter() - started
        beta = mean = scale = None
        fit_seconds = base_seconds = 0.0
        correction_key = None
        if X_fit.cov_names:
            stage = time.perf_counter()
            base = self._base_prediction(key, calibration)
            base_seconds = time.perf_counter() - stage
            idx = calibration.idx
            steps = np.asarray(HORIZONS, dtype=np.int64)
            future = idx[:, None] + steps
            if np.any(future >= len(calibration.r1)) or not np.all(
                    calibration.ts[future] == calibration.ts[idx, None] + steps * 60):
                raise ValueError("Residual labels require contiguous minute timestamps")
            target = np.cumsum(calibration.r1[future], axis=1)
            residual = np.asarray(target - base, dtype=np.float64)
            x = self._design(calibration, base)
            if not np.isfinite(residual).all():
                raise ValueError("Nonfinite residual labels")
            if len(X_pred.idx) and calibration.ts[future].max() >= X_pred.ts[X_pred.idx].min():
                raise ValueError("Residual training labels must precede all prediction origins")
            correction_key = hashlib.sha256((key + digest(x, residual) + repr(X_fit.cov_names)).encode()).hexdigest()
            stage = time.perf_counter()
            mean, scale = x.mean(axis=0), x.std(axis=0)
            scale[scale < 1e-12] = 1.0
            a = torch.as_tensor(np.column_stack((np.ones(len(x)), (x - mean) / scale)),
                                dtype=torch.float64, device="cuda")
            y = torch.as_tensor(residual, dtype=torch.float64, device="cuda")
            ridge = float(self.residual["ridge"])
            if ridge:
                penalty = torch.eye(a.shape[1], dtype=a.dtype, device=a.device) * ridge ** 0.5
                penalty[0, 0] = 0  # never penalize the intercept
                a = torch.cat((a, penalty))
                y = torch.cat((y, torch.zeros((penalty.shape[0], len(HORIZONS)), dtype=y.dtype, device=y.device)))
            # Multiple RHS are independent horizon heads, sharing the design/SVD only.
            beta = torch.linalg.pinv(a, rtol=float(self.residual["pinv_rtol"])) @ y
            torch.cuda.synchronize()
            if not torch.isfinite(beta).all().item():
                raise ValueError("Nonfinite GPU residual coefficients")
            fit_seconds = time.perf_counter() - stage
            if self.adapter_dir:
                atomic_json(self.adapter_dir / "residual_heads" / f"{correction_key}.json",
                            {"contract": CONTRACT, "adapter": key, "data": grid_digest(calibration),
                             "split": split, "features": list(X_fit.cov_names), "residual": self.residual,
                             "mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.cpu().tolist(),
                             "target": "cumulative_log_return_minus_tfm", "device": "cuda"})
        names = tuple(X_fit.cov_names)

        def predict(seq):
            if tuple(seq.cov_names) != names:
                raise ValueError("Residual predictor feature order differs from fit")
            base = self._base_prediction(key, seq)
            if beta is None:
                return base.copy()
            x = self._design(seq, base)
            result = np.empty_like(base)
            for start in range(0, len(x), self.batch_size):
                stop = start + self.batch_size
                a = torch.as_tensor(np.column_stack((np.ones(len(x[start:stop])),
                                                     (x[start:stop] - mean) / scale)),
                                    dtype=torch.float64, device="cuda")
                result[start:stop] = base[start:stop] + (a @ beta).cpu().numpy()
            return result

        stage = time.perf_counter()
        prediction = predict(X_pred)
        prediction_seconds = time.perf_counter() - stage
        if self.adapter_dir:
            record = {"adapter": key, "residual_head": correction_key, "features": list(names),
                      "split": split, "lora_or_cache_seconds": lora_seconds,
                      "calibration_forecast_or_cache_seconds": base_seconds, "residual_fit_seconds": fit_seconds,
                      "prediction_or_cache_seconds": prediction_seconds, "n_predict": len(X_pred.idx)}
            atomic_json(self.adapter_dir / "runtime" / f"{uuid.uuid4().hex}.json", record)
        e = int(meta["best_epoch"])
        return FitResult(prediction, (e, e, e), [predict], is_logret=True)

    def artifact_meta(self, covariates=(), native=True):
        result = super().artifact_meta(covariates, native)
        result["xreg"] = None if not covariates else {
            "mode": "tfm_forecast_then_heldout_residual", "input": "features_at_origin_and_tfm_vector",
            "one_origin_per_call": False, "fit_per": "fold/feature-set/seed; independent horizon heads",
            "settings": self.residual, "device": "cuda", "contract": CONTRACT}
        result["calibration"] = "Chronological FIT prefix LoRA / inner ES / held-out residual suffix; outer ES unused"
        return result
