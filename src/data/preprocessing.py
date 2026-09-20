from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from .ofi import order_flow
from src.models import WINDOW_LOCAL_NORMALIZATION

RAW_COLUMNS = [f"{side}_{field}_{level}" for side in ("bid", "ask")
               for field in ("price", "qty") for level in range(1, 11)]

TRAIN_GLOBAL_NORMALIZATION = "per-field train-only z-score; no target scaling"

# HFformer is the single exception: it never sees a corpus-level statistic.
GLOBAL_STANDARDIZER_MODELS = ("e0", "ofi_lstm", "patchtst", "moderntcn", "lit")


# How each policy is evaluated numerically; the policy string itself stays frozen.
NORMALIZATION_IMPLEMENTATION = {
    TRAIN_GLOBAL_NORMALIZATION: "float64 train-split fit and transform; float32 model input",
    WINDOW_LOCAL_NORMALIZATION: ("inside HFformer.forward; float64 accumulation for the window "
                                 "reduction, float32 result, so the z-score depends on the sample's "
                                 "own history alone and never on the batch"),
}


def normalization_policy(model):
    return TRAIN_GLOBAL_NORMALIZATION if model in GLOBAL_STANDARDIZER_MODELS else WINDOW_LOCAL_NORMALIZATION


@dataclass
class RawBook:
    timestamps: np.ndarray  # UTC integer nanoseconds, never float epoch seconds.
    book: np.ndarray        # [N, 2 sides, 10 levels, 2 fields], float64.
    mid: np.ndarray
    segments: np.ndarray
    bad_edges: np.ndarray
    bad_prefix: np.ndarray
    stats: dict
    source: dict
    column_mapping: dict


def load_csv(config):
    path = Path(config.csv_path).expanduser().resolve()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    mapping = {name: config.column_mapping.get(name, name) for name in RAW_COLUMNS}
    if len(set(mapping.values())) != 40:
        raise ValueError("Column mapping must be one-to-one")
    required = [config.timestamp_column, *mapping.values()]
    missing = set(required)-set(header)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    segment = config.segment_column if config.segment_column in header else None
    df = pd.read_csv(path, usecols=required+([segment] if segment else []))
    if len(df) < 3 or df.isna().any().any():
        raise ValueError("CSV must contain >=3 complete, finite snapshots")
    stamp = df[config.timestamp_column]
    if pd.api.types.is_numeric_dtype(stamp):
        parsed = pd.to_datetime(stamp, unit=config.timestamp_unit, utc=True)
    else:
        parsed = pd.to_datetime(stamp, utc=True)
    # Explicit ns conversion also handles pandas 3's inferred datetime resolution.
    ts = np.ascontiguousarray(parsed.dt.as_unit("ns").astype("int64").to_numpy())
    dt = np.diff(ts)/1e9
    if np.any(dt <= 0):
        raise ValueError("Timestamps must be strictly increasing; do not silently reorder/deduplicate")
    values = df[[mapping[c] for c in RAW_COLUMNS]].to_numpy(dtype=np.float64)
    book = np.ascontiguousarray(values.reshape(-1, 2, 2, 10).transpose(0, 1, 3, 2))
    if not np.isfinite(book).all() or np.any(book[..., 0] <= 0) or np.any(book[..., 1] < 0):
        raise ValueError("Prices must be positive; quantities nonnegative; all values finite")
    if np.any(book[:, 0, 0, 0] > book[:, 1, 0, 0]):
        raise ValueError("Crossed book detected")
    if np.any(np.diff(book[:, 0, :, 0], axis=1) > 0) or np.any(np.diff(book[:, 1, :, 0], axis=1) < 0):
        raise ValueError("Depth prices are out of order; verify column mapping")
    segments = pd.factorize(df[segment], sort=False)[0] if segment else np.zeros(len(ts), dtype=np.int64)
    bad = (dt > config.max_gap_seconds) | (segments[1:] != segments[:-1])
    prefix = np.r_[0, np.cumsum(bad, dtype=np.int64)]
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    stats = dict(rows=len(ts), median_dt_seconds=float(np.median(dt)),
                 p99_dt_seconds=float(np.quantile(dt, .99)), max_dt_seconds=float(dt.max()),
                 gaps_gt_2_seconds=int((dt > 2).sum()),
                 segment_transitions=int((segments[1:] != segments[:-1]).sum()))
    return RawBook(ts, book, np.ascontiguousarray(book[:, :, 0, 0].mean(1)),
                   segments, bad, prefix, stats,
                   dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size),
                   dict(features=mapping, timestamp=config.timestamp_column,
                        timestamp_unit=config.timestamp_unit, segment=segment))


def chronological_split(raw, config):
    ts = raw.timestamps
    if config.train_end:
        cut1 = int(pd.Timestamp(config.train_end).value)
        cut2 = int(pd.Timestamp(config.validation_end).value)
    else:
        duration = int(ts[-1])-int(ts[0])
        cut1 = int(ts[0])+round(duration*config.train_fraction)
        cut2 = int(ts[0])+round(duration*(config.train_fraction+config.validation_fraction))
    if not ts[0] < cut1 < cut2 <= ts[-1]:
        raise ValueError("Split timestamps must be ordered inside the dataset")
    i, j = (int(np.searchsorted(ts, cut, side="left")) for cut in (cut1, cut2))
    ranges = dict(train=(0, i), validation=(i, j), test=(j, len(ts)))
    if any(hi-lo < 2 for lo, hi in ranges.values()):
        raise ValueError("Empty/tiny timestamp split; configure explicit train_end/validation_end")
    manifest = dict(policy="strict: history, origin, and all targets inside one split",
                    source=raw.source, boundaries_ns=[cut1, cut2], ranges=ranges,
                    boundaries_utc=[pd.Timestamp(t, unit="ns", tz="UTC").isoformat() for t in (cut1, cut2)])
    return ranges, manifest


def features(raw, model, representation, ranges):
    reset = np.r_[True, raw.bad_edges].copy()
    for lo, _ in ranges.values():
        reset[lo] = True
    book = raw.book
    if model == "ofi_lstm":
        x = order_flow(book, reset, representation)
        names = ([f"{side}_OF_{i}" for side in ("bid", "ask") for i in range(1, 11)]
                 if representation == "of" else [f"OFI_{i}" for i in range(1, 11)])
        layout = ["time", "feature"]
    elif model == "hfformer":
        raw9 = book[:, :, :9, :].transpose(0, 1, 3, 2).reshape(-1, 36)
        lag = np.r_[0., np.diff(np.log(raw.mid))]
        lag[reset] = 0
        bp, ap = book[:, 0, 0, 0], book[:, 1, 0, 0]
        bq, aq = book[:, 0, 0, 1], book[:, 1, 0, 1]
        weighted = np.divide(ap*bq+bp*aq, bq+aq, out=raw.mid.copy(), where=(bq+aq)>0)
        x = np.column_stack([raw9, lag, weighted])
        names = [f"{s}_{f}_{i}" for s in ("bid", "ask") for f in ("price", "qty") for i in range(1, 10)]
        names += ["lagged_log_return_1_snapshot", "weighted_mid_price_L1"]
        layout = ["time", "feature"]
    elif model == "lit":
        x = book
        names = [f"{s}_{f}_{i}" for s in ("bid", "ask") for i in range(1, 11) for f in ("price", "qty")]
        layout = ["time", "side:bid,ask", "depth:1..10", "field:price,qty"]
    else:
        x = book.transpose(0, 1, 3, 2).reshape(-1, 40)
        names, layout = RAW_COLUMNS, ["time", "feature"]
    policy = normalization_policy(model)
    schema = dict(model=model, names=names, layout=layout, sample_shape=list(x.shape[1:]),
                  normalization=policy,
                  normalization_implementation=NORMALIZATION_IMPLEMENTATION[policy],
                  reset_policy="zero lag/flow at first row, gap, segment and split boundary",
                  weighted_mid_definition="(ask_price_1*bid_qty_1 + bid_price_1*ask_qty_1)/(bid_qty_1+ask_qty_1); zero-volume fallback=mid")
    return np.ascontiguousarray(x), schema


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x):
        # Accumulate float64 before casting prices to float32 (preserves small moves).
        mean, scale = x.mean(axis=0, dtype=np.float64), x.std(axis=0, dtype=np.float64)
        return cls(mean, np.where(scale < 1e-12, 1., scale))

    def transform(self, x):
        return np.ascontiguousarray((x-self.mean)/self.scale, dtype=np.float32)

    def to_dict(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), fitted_on="train_only")

    @classmethod
    def from_dict(cls, obj):
        mean, scale = np.asarray(obj["mean"]), np.asarray(obj["scale"])
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid saved standardizer")
        return cls(mean, scale)
