"""Frozen class definitions for USD mid-price displacement.

Every definition is fitted once, from the fold-1 TRAIN displacements alone, and then
applied unchanged to every split of every fold. Nothing here ever sees validation or
test data: `fit_*` take one array, and scripts/cls_labels.py hands them the fold-1
train samples only. scripts/cls_audit.py refits from the raw file and requires an
exact match with the frozen JSON.

Interval convention: every class is a half-open interval [a, b) on delta, so a
displacement that sits exactly on a boundary belongs to the class above it. The one
exception is the equal-width method's top finite edge +q, which belongs to the last
finite bin, so that the finite range is the closed interval [-q, +q] and only
delta > +q is upper overflow (delta < -q is lower overflow).
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("equal_width", "quantile")
# Mids are (bid+ask)/2 on a 0.05 USD grid, but their float64 difference carries ~1e-11 of
# rounding noise, so one USD displacement has several binary representations (e.g.
# -0.10000000000582 and -0.09999999999127). Every displacement is snapped to 1e-6 USD
# before it is fitted or assigned, so a class is a function of the USD value alone.
DELTA_DECIMALS = 6


def snap(delta):
    return np.round(np.asarray(delta, dtype=np.float64), DELTA_DECIMALS)


def displacement(target_mid, origin_mid):
    """delta = target_mid - origin_mid in USD, snapped to 1e-6 USD."""
    return snap(np.asarray(target_mid, dtype=np.float64)-np.asarray(origin_mid, dtype=np.float64))


@dataclass
class LabelSpec:
    method: str                 # equal_width | quantile
    horizon_seconds: int
    edges: list                 # ascending class boundaries (len = n_classes - 1)
    representatives: list       # decoded displacement per class, USD
    n_classes: int
    top_edge_closed: bool       # equal_width: +q belongs to the last finite bin
    params: dict                # method parameters, e.g. q, K, width, levels
    fit: dict = field(default_factory=dict)  # provenance and fold-1-train statistics

    def assign(self, delta):
        """Deterministic interval assignment, never nearest-representative."""
        delta = snap(delta)
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite displacement")
        edges = np.asarray(self.edges, dtype=np.float64)
        classes = np.searchsorted(edges, delta, side="right")
        if self.top_edge_closed:
            classes[delta == edges[-1]] = len(edges)-1
        return classes.astype(np.int64)

    def decode(self, classes):
        return np.asarray(self.representatives, dtype=np.float64)[np.asarray(classes)]

    def expected(self, probabilities):
        """Secondary diagnostic only: sum_k p_k * representative_k."""
        return np.asarray(probabilities, dtype=np.float64) @ np.asarray(self.representatives, dtype=np.float64)

    def class_intervals(self):
        """Human-readable interval per class, for reports."""
        e = self.edges
        out = []
        last = self.n_classes-1
        for k in range(self.n_classes):
            if k == 0:
                out.append(f"(-inf, {e[0]!r})")
            elif k == last:
                out.append(f"({e[-1]!r}, +inf)" if self.top_edge_closed else f"[{e[-1]!r}, +inf)")
            else:
                closed = self.top_edge_closed and k == last-1
                out.append(f"[{e[k-1]!r}, {e[k]!r}{']' if closed else ')'}")
        return out

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        return cls(**obj)


def _counts(spec, delta):
    classes = spec.assign(delta)
    counts = np.bincount(classes, minlength=spec.n_classes)
    return counts


def distribution(spec, delta):
    """Per-class counts and fractions of any split, for the label-distribution report."""
    counts = _counts(spec, delta)
    n = int(counts.sum())
    return dict(samples=n, counts=counts.tolist(), fractions=(counts/max(n, 1)).tolist(),
                lower_outer_fraction=float(counts[0]/max(n, 1)),
                upper_outer_fraction=float(counts[-1]/max(n, 1)))


def fit_equal_width(delta, horizon_seconds, bins=32, percentile=99.0,
                    overflow="edge_plus_half_width", fitted_on=None):
    """METHOD A: K equal-width bins on [-q, +q], q = percentile of |delta| on F1 train.

    K must be even so that 0 USD is exactly an edge and no finite bin crosses zero.
    Overflow classes: delta < -q and delta > +q, never clipped into the edge bins.
    Overflow representatives: -q - width/2 and +q + width/2 (the official rule);
    `overflow="conditional_median"` is a follow-up variant that uses the fold-1-train
    median displacement of each overflow class instead.
    """
    delta = snap(delta)
    if bins < 2 or bins % 2:
        raise ValueError("equal_width needs an even number of finite bins >= 2")
    q = float(np.percentile(np.abs(delta), percentile))  # numpy default: linear interpolation
    if not q > 0:
        raise ValueError("Degenerate displacement range")
    width = 2*q/bins
    edges = width*np.arange(-(bins//2), bins//2+1, dtype=np.float64)
    edges[0], edges[-1] = -q, q
    if edges[bins//2] != 0.0:
        raise AssertionError("0 USD must be an exact class boundary")
    midpoints = (edges[:-1]+edges[1:])/2
    spec = LabelSpec(method="equal_width", horizon_seconds=int(horizon_seconds),
                     edges=edges.tolist(), representatives=[], n_classes=bins+2,
                     top_edge_closed=True,
                     params=dict(percentile=float(percentile), q=q, bins=int(bins), width=width,
                                 finite_min=-q, finite_max=q, overflow=overflow,
                                 percentile_method="numpy.percentile(|delta|, linear)",
                                 delta_rounding=f"numpy.round(delta, {DELTA_DECIMALS})"),
                     fit={})
    classes = spec.assign(delta)
    if overflow == "edge_plus_half_width":
        low, high = -q-width/2, q+width/2
    elif overflow == "conditional_median":
        low = float(np.median(delta[classes == 0])) if (classes == 0).any() else -q-width/2
        high = float(np.median(delta[classes == bins+1])) if (classes == bins+1).any() else q+width/2
    else:
        raise ValueError(f"Unknown overflow representation {overflow}")
    spec.representatives = [float(low), *midpoints.tolist(), float(high)]
    spec.fit = _fit_record(spec, delta, fitted_on)
    return spec


def fit_quantile(delta, horizon_seconds, n_classes=34, fitted_on=None):
    """METHOD B: equal-frequency bins at the F1-train quantiles of delta.

    Edges are the quantiles at levels j/n_classes, j=1..n_classes-1. A point mass
    (delta == 0 exactly is common at 60 s) can make neighbouring quantiles equal; equal
    edges are merged, so the realized class count can fall below the requested one,
    and the merge is recorded. Interior classes decode to their midpoint; the two
    unbounded outer classes decode to the fold-1-train median of their own samples.
    """
    delta = snap(delta)
    levels = np.arange(1, n_classes)/n_classes
    raw_edges = np.quantile(delta, levels)  # numpy default: linear interpolation
    # Snapped like the data, so an edge that lands on a price-grid point mass is exactly
    # that grid value and the whole mass goes to the class above it.
    edges = np.unique(snap(raw_edges))
    spec = LabelSpec(method="quantile", horizon_seconds=int(horizon_seconds),
                     edges=edges.tolist(), representatives=[], n_classes=len(edges)+1,
                     top_edge_closed=False,
                     params=dict(requested_classes=int(n_classes), levels=levels.tolist(),
                                 raw_quantile_edges=raw_edges.tolist(),
                                 merged_duplicate_edges=int(len(raw_edges)-len(edges)),
                                 quantile_method="numpy.quantile(delta, linear)",
                                 delta_rounding=f"numpy.round(delta, {DELTA_DECIMALS})",
                                 outer_representative="fold-1-train median of the class's own samples",
                                 interior_representative="interval midpoint (a+b)/2"),
                     fit={})
    classes = spec.assign(delta)
    last = spec.n_classes-1
    if not (classes == 0).any() or not (classes == last).any():
        raise ValueError("Empty outer quantile class")
    reps = [float(np.median(delta[classes == 0]))]
    reps += ((edges[:-1]+edges[1:])/2).tolist()
    reps.append(float(np.median(delta[classes == last])))
    spec.representatives = reps
    spec.fit = _fit_record(spec, delta, fitted_on)
    return spec


def _fit_record(spec, delta, fitted_on):
    counts = _counts(spec, delta)
    n = int(len(delta))
    record = dict(fitted_on=fitted_on or {}, samples=n,
                  counts=counts.tolist(), fractions=(counts/n).tolist(),
                  lower_outer_fraction=float(counts[0]/n), upper_outer_fraction=float(counts[-1]/n),
                  delta_sha256=hashlib.sha256(np.ascontiguousarray(delta).tobytes()).hexdigest(),
                  delta_summary=dict(mean=float(delta.mean()), std=float(delta.std()),
                                     min=float(delta.min()), max=float(delta.max()),
                                     zero_fraction=float((delta == 0).mean())))
    if spec.method == "equal_width":
        record["lower_overflow_percent"] = 100*record["lower_outer_fraction"]
        record["upper_overflow_percent"] = 100*record["upper_outer_fraction"]
    return record


class LabelSet:
    """The frozen per-horizon definitions a run uses, as one JSON artifact."""
    def __init__(self, name, specs, meta=None):
        self.name = name
        self.specs = {int(s.horizon_seconds): s for s in specs}
        self.meta = meta or {}

    def __getitem__(self, horizon):
        return self.specs[int(horizon)]

    def to_dict(self):
        return dict(name=self.name, meta=self.meta,
                    horizons={str(h): s.to_dict() for h, s in sorted(self.specs.items())})

    def sha256(self):
        """Fingerprint of the definitions alone (edges, representatives, class count)."""
        core = {str(h): dict(method=s.method, edges=s.edges, representatives=s.representatives,
                             n_classes=s.n_classes, top_edge_closed=s.top_edge_closed)
                for h, s in sorted(self.specs.items())}
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = self.to_dict() | {"definition_sha256": self.sha256()}
        path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
        return path

    @classmethod
    def from_dict(cls, obj):
        specs = [LabelSpec.from_dict(v) for _, v in sorted(obj["horizons"].items(), key=lambda kv: int(kv[0]))]
        label_set = cls(obj["name"], specs, obj.get("meta"))
        if "definition_sha256" in obj and obj["definition_sha256"] != label_set.sha256():
            raise ValueError(f"Label definition {obj['name']} fails its own fingerprint")
        return label_set

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
