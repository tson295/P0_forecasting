"""Time actual evaluation predictions; no benchmark pass, warmup or extra inference."""
from __future__ import annotations

import csv
from array import array
from time import perf_counter

import numpy as np

from .config import write_json

LATENCY_SCOPE = "prepared input/context + scaling + transfers + predict + host log-return output"


def latency_stats(duration_ms, n_predictions, single_origin_sample):
    """Quantiles use individual observed durations, never averages of fold quantiles."""
    durations = np.asarray(duration_ms, np.float64)
    single = durations[np.asarray(single_origin_sample, dtype=bool)]
    count = int(np.asarray(n_predictions, dtype=np.int64).sum())
    return {
        "inference_n_predictions": count,
        "inference_n_calls": len(durations),
        "inference_single_n": len(single),
        "inference_mean_ms": float(single.mean()),
        "inference_p50_ms": float(np.percentile(single, 50)),
        "inference_p95_ms": float(np.percentile(single, 95)),
        "inference_p99_ms": float(np.percentile(single, 99)),
        "inference_upper_bound_ms": float(single.max()),
        "inference_upper_bound_kind": "observed_max_not_hard_bound",
        "inference_first_call_ms": float(durations[0]),
        "inference_batch_p95_ms": float(np.percentile(durations, 95)),
        "inference_batch_p99_ms": float(np.percentile(durations, 99)),
        "inference_batch_max_ms": float(durations.max()),
        "inference_total_seconds": float(durations.sum() / 1000),
        "inference_amortized_ms_per_prediction": float(durations.sum() / count),
        "inference_scope": LATENCY_SCOPE,
        "inference_warmup_calls": 0,
    }


def infer(cfg, ids, out, predict, batch_size=None):
    """Selected evenly spaced origins run as batch 1; their outputs count in metrics.

    Remaining origins use normal batches. Every origin is predicted exactly once.
    p95/p99 refer to sampled batch-1 requests, not batch time divided by batch size.
    """
    import torch

    options = cfg["inference"]
    batch_size = batch_size or options["batch_size"]
    selected = np.unique(np.linspace(0, len(ids) - 1,
                                    min(len(ids), options["single_origin_samples"]), dtype=np.int64))
    selected_set = set(selected.tolist())
    durations, counts, samples = array("d"), array("I"), array("B")
    predictions = np.empty(len(ids), dtype=np.float64)
    fields = ["start_position", "origin_id", "n_predictions", "single_origin_sample", "duration_ms"]
    with (out / "inference_latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        start = 0
        while start < len(ids):
            sampled = start in selected_set
            after = int(np.searchsorted(selected, start, side="right"))
            next_single = int(selected[after]) if after < len(selected) else len(ids)
            stop = start + 1 if sampled else min(start + batch_size, next_single, len(ids))
            # Drain queued training/previous work before the clock; wait for this
            # request's GPU completion before stopping it (CUDA launches are async).
            torch.cuda.synchronize()
            begin = perf_counter()
            result = np.asarray(predict(ids[start:stop]), np.float64).reshape(-1)
            torch.cuda.synchronize()
            elapsed = (perf_counter() - begin) * 1000
            if len(result) != stop - start:
                raise ValueError("Inference output count does not match evaluation origins.")
            row = dict(start_position=start, origin_id=int(ids[start]), n_predictions=stop - start,
                       single_origin_sample=int(sampled), duration_ms=elapsed)
            writer.writerow(row)
            durations.append(elapsed)
            counts.append(stop - start)
            samples.append(int(sampled))
            predictions[start:stop] = result
            start = stop
    stats = latency_stats(durations, counts, samples)
    stats["inference_configured_batch_size"] = batch_size
    write_json(out / "latency.json", stats)
    return predictions, stats
