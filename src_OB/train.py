"""One model per (family, fold, horizon); Vast-only entrypoint, no feature selection."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json
from .data import Data, price_metrics, date_string, select_origins
from .results import gains_vs_e0, atomic_csv, refresh_summaries

FAMILIES = ("lgbm", "xgb", "cat", "xgbrf", "lstm", "autots", "tfm_zero_shot", "tfm_lora")


def resume_cell(cfg, data, output, out):
    """--resume: True for a cell completed with this exact config/revision (kept and skipped). An unfinished
    attempt is moved with its run/failed files to <output>/attempts/<fold>/<model>/<h>/attemptN, never deleted."""
    run = json.loads((out / "run.json").read_text()) if (out / "run.json").is_file() else {}
    if (out / "completed.json").is_file():
        if (run.get("config") != json.loads(json.dumps(cfg))
                or run.get("dataset_revision") != data.meta["dataset_revision"]):
            raise ValueError(f"{out}: completed cell thuộc config/revision khác; không trộn run.")
        return True
    base = output / "attempts" / out.relative_to(output)
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"attempt{n}").exists():
        n += 1
    out.rename(base / f"attempt{n}")
    print(f"moved unfinished attempt {out} -> {base / f'attempt{n}'}", flush=True)
    return False


def train(cfg, models=None, fold_names=None, resume=False):
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
    # Code provenance of this process, taken once before the lazy model imports (checker R6-I3).
    from .prepare import code_provenance
    train_code = code_provenance(cfg)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for fold in data.folds():
        if fold_names and fold.name not in fold_names:
            continue
        # Common evaluation population across native price-sequence and OF models.
        # No model silently scores an easier subset when its context has a gap.
        train_ids, val_ids = select_origins(cfg, data, fold)
        if not len(train_ids) or not len(val_ids):
            raise ValueError(f"{fold.name}: không còn sample với nhãn/context hợp lệ.")
        # Same existing E0: zero return means predicted raw price equals origin mid.
        # Compute once per fold/horizon, reuse for every model on the common VAL ids.
        evaluation = {}
        for horizon in cfg["horizons_seconds"]:
            _, actual = data.target(val_ids, horizon)
            evaluation[horizon] = (actual, price_metrics(actual, data.mid[val_ids]))
        x_train = None
        for model in models:
            if model not in ("lgbm", "xgb", "cat", "xgbrf"):
                x_train = None
            if model in ("lgbm", "xgb", "cat", "xgbrf") and x_train is None:
                from .trees import matrix
                x_train = matrix(data, train_ids)
            for horizon in cfg["horizons_seconds"]:
                out = output / fold.name / model / f"h{horizon}s"
                if resume and out.exists() and resume_cell(cfg, data, output, out):
                    print(f"{fold.name} {model} h={horizon}s completed with this config; skipped (--resume)", flush=True)
                    continue
                out.mkdir(parents=True, exist_ok=False)  # protect prior model/prediction artifacts
                prior = sorted(p.relative_to(output).as_posix()
                               for p in (output / "attempts" / out.relative_to(output)).glob("attempt*"))
                started = time.time()
                print(f"{fold.name} {model} h={horizon}s train={len(train_ids)} val={len(val_ids)}", flush=True)
                write_json(out / "run.json", {"status": "started", "config": cfg, "fold": asdict(fold),
                           "model": model, "horizon_seconds": horizon, "strategy": "direct",
                           "observed_coverage": data.meta["coverage"], "dataset_revision": data.meta["dataset_revision"],
                           # Provenance of the prepared data (replay/code/config) and of the training code.
                           "prepared": {k: data.meta.get(k) for k in
                                        ("replay_version", "code_commit", "code_uncommitted_paths", "config_sha256")},
                           "train_code": train_code, "prior_attempts": prior,
                           "created_at": started, "n_train": len(train_ids), "n_val": len(val_ids)})
                y_train, _ = data.target(train_ids, horizon)
                actual, e0 = evaluation[horizon]
                try:
                    if model in ("lgbm", "xgb", "cat", "xgbrf"):
                        from .trees import run
                        delta, latency = run(model, cfg, x_train, y_train, data, val_ids, out)
                    elif model == "lstm":
                        from .neural import run
                        delta, latency = run(model, cfg, data, train_ids, val_ids, horizon, out)
                    elif model == "autots":
                        from .autots_native import run
                        delta, latency = run(cfg, data, fold, val_ids, horizon, out)
                    else:
                        from .timesfm import run
                        delta, latency = run(model, cfg, data, train_ids, val_ids, horizon, out)
                    predicted = np.asarray(data.mid[val_ids], np.float64) * np.exp(np.asarray(delta, np.float64))
                    metrics = price_metrics(actual, predicted)
                    metrics.update(gains_vs_e0(metrics, e0))
                    pd.DataFrame({"timestamp_us": data.ts[val_ids], "origin_price": data.mid[val_ids],
                                  "horizon_seconds": horizon, "actual_price": actual, "predicted_price": predicted,
                                  "predicted_log_return": delta}).to_parquet(out / "predictions.parquet", index=False)
                    record = {"model": model, "fold": fold.name,
                               "horizon_seconds": horizon, "price": "L2 mid", **metrics,
                               "E0": e0, **latency,
                               "train_end": date_string(fold.train_end), "val_start": date_string(fold.val_start),
                               "gap_days": cfg["gap_days"], "duration_seconds": time.time() - started}
                    write_json(out / "metrics.json", record)
                    atomic_csv(pd.json_normalize(record, sep="_"), out / "metrics.csv")
                    write_json(out / "completed.json", {"status": "completed"})
                    refresh_summaries(output)
                    print(f"{model} {fold.name} h={horizon}s {metrics}", flush=True)
                except BaseException as exc:
                    write_json(out / "failed.json", {"error": str(exc), "type": type(exc).__name__})
                    raise
                finally:
                    torch.cuda.empty_cache()
