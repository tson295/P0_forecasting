"""One model per (family, fold, horizon); Vast-only entrypoint, no feature selection."""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json
from .data import Data, price_metrics, date_string

FAMILIES = ("lgbm", "xgb", "cat", "xgbrf", "lstm", "autots", "tfm_zero_shot", "tfm_lora")


def train(cfg, models=None, fold_names=None):
    # Explicit remote opt-in plus CUDA availability; no fallback or local CPU training.
    if os.environ.get("P0_OB_VAST") != "1":
        raise RuntimeError("Training chỉ trên Vast: đặt P0_OB_VAST=1 trong máy Vast.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Không có CUDA GPU; không chạy training CPU.")
    models = models or cfg["models"]
    if set(models) - set(FAMILIES):
        raise ValueError(f"Model hợp lệ: {FAMILIES}")
    data = Data(cfg)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for fold in data.folds():
        if fold_names and fold.name not in fold_names:
            continue
        train_ids = data.indices(fold.train_start, fold.train_end)
        train_ids = train_ids[data.ts[train_ids - cfg["context"] + 1] >= fold.train_start]
        val_ids = data.indices(fold.val_start, fold.val_end)
        # Common evaluation population across native price-sequence and OF models.
        # No model silently scores an easier subset when its context has a gap.
        population_models = cfg["models"]  # --models splits jobs without changing their scoring population
        if any(m.startswith("tfm") or m == "autots" for m in population_models):
            for horizon in cfg["horizons_seconds"]:
                length = max(cfg["tfm"]["context"] if any(m.startswith("tfm") for m in population_models) else 1,
                             cfg["autots"]["max_window_size"] if "autots" in population_models else 1)
                def eligible(ids):
                    keep = []
                    for s in range(0, len(ids), 4096):
                        batch = ids[s:s + 4096]
                        _, valid = data.price_context(batch, horizon, length)
                        keep.append(batch[valid])
                    return np.concatenate(keep) if keep else np.empty(0, np.int64)
                train_ids, val_ids = eligible(train_ids), eligible(val_ids)
                train_ids = train_ids[data.ts[train_ids] - (length - 1) * horizon * 1_000_000 >= fold.train_start]
        if not len(train_ids) or not len(val_ids):
            raise ValueError(f"{fold.name}: không còn sample với nhãn/context hợp lệ.")
        x_train = x_val = None
        for model in models:
            if model not in ("lgbm", "xgb", "cat", "xgbrf"):
                x_train = x_val = None
            if model in ("lgbm", "xgb", "cat", "xgbrf") and x_train is None:
                from .trees import matrix
                x_train, x_val = matrix(data, train_ids), matrix(data, val_ids)
            for horizon in cfg["horizons_seconds"]:
                out = output / fold.name / model / f"h{horizon}s"
                out.mkdir(parents=True, exist_ok=False)  # protect prior model/prediction artifacts
                started = time.time()
                print(f"{fold.name} {model} h={horizon}s train={len(train_ids)} val={len(val_ids)}", flush=True)
                write_json(out / "run.json", {"status": "started", "config": cfg, "fold": asdict(fold),
                           "model": model, "horizon_seconds": horizon, "strategy": "direct",
                           "created_at": started, "n_train": len(train_ids), "n_val": len(val_ids)})
                y_train, _ = data.target(train_ids, horizon)
                _, actual = data.target(val_ids, horizon)
                try:
                    if model in ("lgbm", "xgb", "cat", "xgbrf"):
                        from .trees import run
                        delta = run(model, cfg, x_train, y_train, x_val, out)
                    elif model == "lstm":
                        from .neural import run
                        delta = run(model, cfg, data, train_ids, val_ids, horizon, out)
                    elif model == "autots":
                        from .autots_native import run
                        delta = run(cfg, data, fold, val_ids, horizon, out)
                    else:
                        from .timesfm import run
                        delta = run(model, cfg, data, train_ids, val_ids, horizon, out)
                    predicted = np.asarray(data.mid[val_ids], np.float64) * np.exp(np.asarray(delta, np.float64))
                    metrics = price_metrics(actual, predicted)
                    pd.DataFrame({"timestamp_us": data.ts[val_ids], "origin_price": data.mid[val_ids],
                                  "horizon_seconds": horizon, "actual_price": actual, "predicted_price": predicted,
                                  "predicted_log_return": delta}).to_parquet(out / "predictions.parquet", index=False)
                    write_json(out / "metrics.json", {"model": model, "fold": fold.name,
                               "horizon_seconds": horizon, "price": "L2 mid", **metrics,
                               "E0": price_metrics(actual, data.mid[val_ids]),
                               "train_end": date_string(fold.train_end), "val_start": date_string(fold.val_start),
                               "gap_days": cfg["gap_days"], "duration_seconds": time.time() - started})
                    write_json(out / "completed.json", {"status": "completed"})
                    print(f"{model} {fold.name} h={horizon}s {metrics}", flush=True)
                except BaseException as exc:
                    write_json(out / "failed.json", {"error": str(exc), "type": type(exc).__name__})
                    raise
                finally:
                    torch.cuda.empty_cache()
