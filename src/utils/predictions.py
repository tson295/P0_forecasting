"""Deterministic prediction export: every figure must be reproducible offline."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.metrics import HORIZON_LABELS, metrics_from_arrays

PREDICTION_COLUMNS = ["origin_index", "origin_timestamp_ns", "origin_mid"] + [
    column.format(h=h) for h in HORIZON_LABELS
    for column in ("target_index_{h}", "target_timestamp_ns_{h}", "target_mid_{h}",
                   "true_return_{h}", "pred_return_{h}", "pred_mid_{h}")]


@torch.no_grad()
def predict(model, dataset, device, batch_size=256, num_workers=0):
    """FP32, eval mode, sequential order: bit-reproducible on the same device."""
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda",
                        drop_last=False,
                        **(dict(persistent_workers=True) if num_workers else {}))
    chunks = []
    for x, _ in loader:
        prediction = model(x.to(device, non_blocking=True))
        if prediction.shape[1:] != (3,):
            raise ValueError(f"Model must emit [B, 3], got {tuple(prediction.shape)}")
        chunks.append(prediction.float().cpu())
    predictions = torch.cat(chunks).numpy().astype(np.float64)
    if len(predictions) != len(dataset):
        raise ValueError("Prediction count does not match the dataset")
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite predictions")
    return predictions


def prediction_frame(dataset, raw, predictions):
    """One row per origin, ascending origin timestamp, with mids for plotting."""
    origins, targets = dataset.origins, dataset.target_indices
    if not (np.diff(raw.timestamps[origins]) > 0).all():
        raise ValueError("Origins must be strictly ordered by timestamp")
    origin_mid = raw.mid[origins].astype(np.float64)
    table = {"origin_index": origins.astype(np.int64),
             "origin_timestamp_ns": raw.timestamps[origins].astype(np.int64),
             "origin_mid": origin_mid}
    for k, h in enumerate(HORIZON_LABELS):
        index = targets[:, k]
        target_mid = raw.mid[index].astype(np.float64)
        table[f"target_index_{h}"] = index.astype(np.int64)
        table[f"target_timestamp_ns_{h}"] = raw.timestamps[index].astype(np.int64)
        table[f"target_mid_{h}"] = target_mid
        table[f"true_return_{h}"] = np.log(target_mid/origin_mid)
        table[f"pred_return_{h}"] = predictions[:, k]
        table[f"pred_mid_{h}"] = origin_mid*np.exp(predictions[:, k])
    frame = pd.DataFrame(table)[PREDICTION_COLUMNS]
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("Nonfinite prediction table")
    return frame


def export_split(model, dataset, raw, device, path, batch_size=256, num_workers=0):
    """Writes <split>_predictions.csv.gz and returns metrics from that exact table."""
    frame = prediction_frame(dataset, raw, predict(model, dataset, device, batch_size, num_workers))
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip container itself byte-reproducible across reruns.
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    truth = frame[[f"true_return_{h}" for h in HORIZON_LABELS]].to_numpy(dtype=np.float64)
    predicted = frame[[f"pred_return_{h}" for h in HORIZON_LABELS]].to_numpy(dtype=np.float64)
    return frame, metrics_from_arrays(predicted, truth)
