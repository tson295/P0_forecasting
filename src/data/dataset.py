"""O(N) arrays and O(number of origins) indices; never store overlapping windows."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (load_csv, chronological_split, features, Standardizer,
                            GLOBAL_STANDARDIZER_MODELS, normalization_policy)


class LOBDataset(Dataset):
    def __init__(self, x, raw, bounds, history_rows, stride_rows, config):
        self.x, self.mid = x, raw.mid
        self.timestamps = raw.timestamps
        self.history_rows = history_rows
        lo, hi = bounds
        origins = np.arange(lo+history_rows-1, hi, stride_rows, dtype=np.int64)
        wanted = raw.timestamps[origins, None] + np.array(config.horizons_seconds, dtype=np.int64)*10**9
        target = np.searchsorted(raw.timestamps, wanted, side="left")
        in_bounds = (target < hi).all(axis=1)
        safe = np.minimum(target, len(raw.timestamps)-1)
        overshoot = raw.timestamps[safe]-wanted
        tolerance = round(config.target_tolerance_seconds*1e9)
        valid = in_bounds & ((overshoot >= 0) & (overshoot <= tolerance)).all(axis=1)
        starts = origins-history_rows+1
        # P[b]-P[a] counts all bad edges along inclusive observations [a,b].
        valid &= raw.bad_prefix[safe[:, -1]]-raw.bad_prefix[starts] == 0
        self.origins = np.ascontiguousarray(origins[valid])
        self.target_indices = np.ascontiguousarray(target[valid])

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        origin = self.origins[idx]
        x = torch.from_numpy(self.x[origin-self.history_rows+1:origin+1])
        y = np.log(self.mid[self.target_indices[idx]]/self.mid[origin]).astype(np.float32)
        return x, torch.from_numpy(y)


@dataclass
class PreparedData:
    raw: object
    datasets: dict
    metadata: dict
    history_rows: int
    channels: int


def prepare_data(config, saved_metadata=None):
    config.validate()
    raw = load_csv(config.data)
    ranges, manifest = chronological_split(raw, config.data)
    train_hi = ranges["train"][1]
    train_dt = np.diff(raw.timestamps[:train_hi])/1e9
    train_good = ~raw.bad_edges[:train_hi-1]
    if not train_good.any():
        raise ValueError("Train split has no continuous sampling intervals")
    median_dt = float(np.median(train_dt[train_good]))
    stride_seconds = config.data.stride_seconds or config.data.history_seconds/6
    history = config.data.history_rows or math.ceil(config.data.history_seconds/median_dt)
    stride = max(1, int(math.floor(stride_seconds/median_dt+0.5)))
    x, schema = features(raw, config.model, config.data.of_representation, ranges)
    # HFformer normalizes inside its own forward pass, from the sample's history
    # alone, so no corpus-level statistic is fitted or stored for it at all.
    global_standardizer = config.model in GLOBAL_STANDARDIZER_MODELS
    if saved_metadata:
        old = saved_metadata["preprocessing"]
        if schema != saved_metadata["feature_schema"]:
            raise ValueError("Feature schema does not match checkpoint")
        if config.data.history_seconds != old["history_seconds"] or config.data.history_rows != old["history_rows_override"]:
            raise ValueError("Checkpoint history configuration differs")
        if bool(old["standardizer"]) != global_standardizer:
            raise ValueError("Checkpoint normalization policy differs")
        # Fine-tuning keeps base feature scaling and shape; no val/test refitting.
        scaler = Standardizer.from_dict(old["standardizer"]) if global_standardizer else None
        history, median_dt = old["history_rows"], old["median_train_dt_seconds"]
        stride = max(1, int(math.floor(stride_seconds/median_dt+0.5)))
    else:
        scaler = Standardizer.fit(x[:train_hi]) if global_standardizer else None
    x = scaler.transform(x) if scaler is not None else np.ascontiguousarray(x, dtype=np.float32)
    datasets = {name: LOBDataset(x, raw, bounds, history, stride, config.data)
                for name, bounds in ranges.items()}
    if any(len(d) == 0 for d in datasets.values()):
        raise ValueError(f"No valid windows in a split: { {k:len(v) for k,v in datasets.items()} }")
    manifest["sample_counts"] = {name: len(ds) for name, ds in datasets.items()}
    manifest["origin_index_sha256"] = {}
    import hashlib
    for name, ds in datasets.items():
        manifest["origin_index_sha256"][name] = hashlib.sha256(ds.origins.tobytes()).hexdigest()
    metadata = dict(
        preprocessing=dict(standardizer=scaler.to_dict() if scaler is not None else None,
                           normalization=normalization_policy(config.model),
                           median_train_dt_seconds=median_dt,
                           history_seconds=config.data.history_seconds, history_rows=history,
                           history_rows_override=config.data.history_rows,
                           history_rounding="ceil(seconds/median_train_continuous_dt)",
                           stride_seconds=stride_seconds, stride_rows=stride,
                           column_mapping=raw.column_mapping),
        feature_schema=schema,
        target_config=dict(horizons_seconds=list(config.data.horizons_seconds),
                           definition="log(mid[target]/mid[origin]); mid=(best_bid+best_ask)/2",
                           lookup="first timestamp >= origin + horizon (searchsorted left)",
                           tolerance_seconds=config.data.target_tolerance_seconds,
                           max_gap_seconds=config.data.max_gap_seconds,
                           segment_boundary_rejection=True),
        split_manifest=manifest, data_stats=raw.stats)
    return PreparedData(raw, datasets, metadata, history, int(np.prod(x.shape[1:])))
