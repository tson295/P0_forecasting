"""Harness: Store (B0 matrix + ext), ColSet, run_config (một configuration × 5 fold → RMSE 15 ô), calibrate (15fixed_m), seed_noise (ε_m).

Bất biến: TargetTransform fit trên FIT của fold (train-only); ES ≠ VAL; metric trên giá sau decode; feature chỉ dùng τ ≤ t.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from Baseline_LGBM import FINE_FEATURE_NAMES, FeatureData, build_lgbm_matrix, build_ohlcv_features, lgbm_feature_names

from .config import HORIZONS
from .transform import TargetTransform  # bản tái hiện đúng công thức B0 (B0 gốc có bug broadcast in-place, xem transform.py)
from .data import grid_frame, to_b0_frame
from .features_ext import ALL_EXT_COLUMNS, compute_ext
from .metrics import cell_metrics, e0_rmse, gain_pp, seed_noise_cells, seed_noise_eps
from .models import FitResult, TabularModel
from .split import Fold


@dataclass(frozen=True)
class ColSet:
    b0: tuple[str, ...]
    ext: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return self.b0 + self.ext

    def with_ext(self, cols) -> "ColSet":
        return ColSet(self.b0, self.ext + tuple(c for c in cols if c not in self.ext))

    def without_ext(self, cols) -> "ColSet":
        drop = set(cols)
        return ColSet(self.b0, tuple(c for c in self.ext if c not in drop))

    def to_dict(self) -> dict:
        return {"b0": list(self.b0), "ext": list(self.ext)}

    @classmethod
    def from_dict(cls, d: dict) -> "ColSet":
        return cls(tuple(d["b0"]), tuple(d.get("ext", ())))

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ColSet":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class Store:
    """Dữ liệu một dataset: FeatureData của B0 (306 cột, eligible, target, rv60), lưới + amount, ext cache."""

    def __init__(self, raw_hf: pd.DataFrame, raw_lf: pd.DataFrame | None = None):
        b0_frame = to_b0_frame(raw_hf)
        self.fd: FeatureData = build_ohlcv_features(b0_frame)
        self.ts = self.fd.frame["timestamp"].to_numpy(np.int64)
        self.eligible = self.fd.eligible
        self.close = self.fd.frame["Close"].to_numpy(float)
        self.grid = grid_frame(self.fd.frame, raw_hf)
        self.raw_lf = raw_lf
        self.b0_names: tuple[str, ...] = lgbm_feature_names()
        self._b0_pos = {n: i for i, n in enumerate(self.b0_names)}
        self._ext: dict[str, np.ndarray] = {}
        self._r1: np.ndarray | None = None

    @property
    def first_origin_ts(self) -> int:
        return int(self.ts[np.flatnonzero(self.eligible)[0]])

    @property
    def last_ts(self) -> int:
        return int(self.ts[-1])

    @property
    def r1(self) -> np.ndarray:
        """Log-return 1 phút trên lưới: r1[s] = log(C_s / C_(s−1)); r1[0] = 0 (không có bar trước; origin đầu ở bar 631)."""
        if self._r1 is None:
            self._r1 = np.concatenate([[0.0], np.diff(np.log(self.close))])
        return self._r1

    def ensure_ext(self, cols) -> None:
        missing = tuple(c for c in cols if c not in self._ext)
        if missing:
            df = compute_ext(self.grid, self.raw_lf, columns=missing)
            for c in missing:
                self._ext[c] = df[c].to_numpy(np.float32)

    def ext_column(self, col: str) -> np.ndarray:
        self.ensure_ext((col,))
        return self._ext[col]

    def all_b0(self) -> ColSet:
        return ColSet(self.b0_names)

    def matrix(self, idx: np.ndarray, colset: ColSet) -> np.ndarray:
        """X (n, |b0| + |ext|) float32 cho các origin idx. Cột B0 từ build_lgbm_matrix (code gốc); ext được phép NaN."""
        parts = []
        if colset.b0:
            x306 = build_lgbm_matrix(self.fd, idx)
            parts.append(x306[:, [self._b0_pos[n] for n in colset.b0]])
        if colset.ext:
            self.ensure_ext(colset.ext)
            parts.append(np.column_stack([self._ext[c][idx] for c in colset.ext]))
        return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((len(idx), 0), np.float32)

    def grid_matrix(self, colset: ColSet) -> np.ndarray:
        """Cột của colset trên TOÀN lưới (bar không eligible = NaN) — regressor theo phút của AutoTS."""
        out = np.full((len(self.ts), len(colset.names)), np.nan, np.float32)
        el = np.flatnonzero(self.eligible)
        out[el] = self.matrix(el, colset)
        return out

    def targets(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_t = self.close[idx]
        c_future = np.column_stack([self.close[idx + h] for h in HORIZONS])
        return c_t, c_future, self.fd.rv60[idx]

    def fine_bases(self, colset: ColSet) -> list[str]:
        bases = []
        for n in colset.b0:
            if n.startswith("fine:"):
                b = n.split(":")[-1]
                if b not in bases:
                    bases.append(b)
        return bases

    def fine_names(self, colset: ColSet) -> list[str]:
        """Tên kênh của ma trận LSTM theo đúng thứ tự cột (dùng để định vị kênh ext khi tính PI)."""
        return [f"fine:{b}" for b in self.fine_bases(colset)] + ["rv60"] + list(colset.ext)

    def fine_matrix(self, colset: ColSet) -> tuple[np.ndarray, list[str]]:
        """Feature theo phút cho LSTM: fine feature B0 còn ≥ 1 cột trong colset.b0 (+ rv60) + ext của colset."""
        cols = [self.fd.fine[:, FINE_FEATURE_NAMES.index(b)] for b in self.fine_bases(colset)]
        cols.append(self.fd.rv60.astype(np.float32))
        for c in colset.ext:
            cols.append(self.ext_column(c))
        return np.column_stack(cols).astype(np.float32), self.fine_names(colset)


@dataclass
class FoldState:
    fold: Fold
    idx_fit: np.ndarray
    idx_es: np.ndarray
    idx_val: np.ndarray
    transform: TargetTransform
    result: FitResult
    X_val: Any
    yhat: np.ndarray  # log-return dự báo trên VAL (n, 3)


@dataclass
class RunResult:
    model: str
    colset: ColSet
    seed: int
    rounds: list[tuple[int, int, int]]
    rmse: np.ndarray  # (F, 3) trên giá
    mae: np.ndarray
    r: np.ndarray
    dir_acc: np.ndarray
    e0: np.ndarray  # (F, 3)
    best_iters: np.ndarray  # (F, 3)
    fold_names: list[str]
    states: list[FoldState] = field(default_factory=list)

    def gain_vs(self, base_rmse: np.ndarray) -> np.ndarray:
        return gain_pp(self.rmse, base_rmse)

    def preds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(s.idx_val, s.yhat) for s in self.states]

    def to_dict(self) -> dict:
        return {
            "model": self.model, "colset": self.colset.to_dict(), "seed": self.seed, "rounds": [list(r) for r in self.rounds],
            "rmse": self.rmse.tolist(), "mae": self.mae.tolist(), "r": self.r.tolist(), "dir_acc": self.dir_acc.tolist(),
            "e0": self.e0.tolist(), "best_iters": self.best_iters.tolist(), "folds": self.fold_names,
        }


def _resolve_rounds(rounds, fold_name: str, supports: bool):
    if rounds is None or not supports:
        return None
    if isinstance(rounds, dict):
        return tuple(int(x) for x in rounds[fold_name])
    return tuple(int(x) for x in rounds)


def _standardize_fit(feats: np.ndarray, idx_fit: np.ndarray) -> np.ndarray:
    mu = np.nanmean(feats[idx_fit], axis=0)
    sd = np.nanstd(feats[idx_fit], axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    z = (feats - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def run_config(store: Store, model: TabularModel, colset: ColSet, folds: list[Fold], rounds=None, seed: int = 8586,
               keep_states: bool = True) -> RunResult:
    """Một configuration (model, colset) trên các fold. rounds: None (ES) | tuple(3) | dict[fold.name → tuple(3)]."""
    F = len(folds)
    rmse = np.zeros((F, 3)); mae = np.zeros((F, 3)); rr = np.zeros((F, 3)); dacc = np.zeros((F, 3)); e0 = np.zeros((F, 3))
    best = np.zeros((F, 3), dtype=int)
    used, states = [], []
    kind = getattr(model, "input_kind", "tabular")
    feats_all = names = None
    if kind == "sequence":
        feats_all, names = store.fine_matrix(colset)
    elif kind == "series":  # covariate/regressor thô theo phút (chuẩn hoá train-only, NaN → 0 như plan §2.3)
        cov_cols = tuple(colset.names) if getattr(model, "series_covariates", "ext") == "all" else tuple(colset.ext)
        if cov_cols:
            feats_all = store.grid_matrix(ColSet(tuple(c for c in cov_cols if c in store._b0_pos),
                                                 tuple(c for c in cov_cols if c not in store._b0_pos)))
            names = cov_cols
    for i, fold in enumerate(folds):
        idx_fit = fold.fit.origins(store.ts, store.eligible)
        idx_es = fold.es.origins(store.ts, store.eligible)
        idx_val = fold.val.origins(store.ts, store.eligible)
        if min(len(idx_fit), len(idx_es), len(idx_val)) == 0:
            raise ValueError(f"{fold.name}: partition rỗng (fit={len(idx_fit)}, es={len(idx_es)}, val={len(idx_val)})")
        transform = TargetTransform.fit(store.fd.target, store.fd.rv60, idx_fit)
        z_fit = transform.encode(store.fd.target, store.fd.rv60, idx_fit)
        z_es = transform.encode(store.fd.target, store.fd.rv60, idx_es)
        fold_rounds = _resolve_rounds(rounds, fold.name, getattr(model, "supports_rounds", True))
        if kind == "sequence":
            from .models_lstm import SeqBatch

            feats = _standardize_fit(feats_all, idx_fit)
            X_fit, X_es, X_val = SeqBatch(feats, idx_fit), SeqBatch(feats, idx_es), SeqBatch(feats, idx_val)
        elif kind == "series":
            from .models import SeriesBatch

            cov = _standardize_fit(feats_all, idx_fit) if feats_all is not None else None
            X_fit, X_es, X_val = (SeriesBatch(store.ts, store.r1, i, cov, tuple(names or ())) for i in (idx_fit, idx_es, idx_val))
        else:
            X_fit, X_es, X_val = store.matrix(idx_fit, colset), store.matrix(idx_es, colset), store.matrix(idx_val, colset)
        res = model.fit_predict(X_fit, z_fit, X_es, z_es, X_val, fold_rounds, seed)
        # TimesFM/AutoTS trả thẳng log-return (§6.7); tree/LSTM trả z-space của B0 → decode với rv60 của đúng origin
        yhat = np.asarray(res.pred_z, np.float32) if res.is_logret else transform.decode(res.pred_z, store.fd.rv60[idx_val])
        c_t, c_future, _ = store.targets(idx_val)
        m = cell_metrics(c_t, c_future, yhat)
        rmse[i], mae[i], rr[i], dacc[i] = m["rmse"], m["mae"], m["r"], m["dir_acc"]
        e0[i] = e0_rmse(c_t, c_future)
        best[i] = res.best_iters
        used.append(tuple(int(x) for x in res.best_iters))
        if keep_states:
            states.append(FoldState(fold, idx_fit, idx_es, idx_val, transform, res, X_val, yhat))
    return RunResult(getattr(model, "name", "?"), colset, seed, used, rmse, mae, rr, dacc, e0, best, [f.name for f in folds], states)


def calibrate(store: Store, model: TabularModel, colset: ColSet, folds: list[Fold], seed: int = 8586, keep_states: bool = True) -> RunResult:
    """Run ES một lần → best_iteration per fold × horizon = số vòng cố định của phase (§1.3)."""
    return run_config(store, model, colset, folds, rounds=None, seed=seed, keep_states=keep_states)


def rounds_from(run: RunResult) -> dict[str, tuple[int, int, int]]:
    return {name: tuple(int(x) for x in run.best_iters[i]) for i, name in enumerate(run.fold_names)}


def seed_noise(store: Store, model: TabularModel, colset: ColSet, folds: list[Fold], rounds, eval_seeds, floor_pp: float = 0.005,
               keep_states_seed: int | None = None) -> tuple[float, np.ndarray, list[RunResult]]:
    """ε (§1.3): chạy CÁC EVALUATION SEED với số vòng cố định (seed ES/calibrate không tham gia).

    Mỗi ô (fold, horizon): mu/sigma của các RMSE → noise_cell = 100·sigma/mu (pp); ε = max(floor, RMS 15 ô).
    Không seed nào được dùng làm mốc/mẫu số. Trả (ε, bảng noise 15 ô, các run).
    """
    if not getattr(model, "seed_dependent", True):
        # inference tất định (TimesFM zero-shot): 3 seed cho kết quả y hệt → chạy MỘT lần, ε = floor,
        # nhiễu seed = 0. KHÔNG tạo ngẫu nhiên nhân tạo để giả calibration (§1.3).
        run = run_config(store, model, colset, folds, rounds=rounds, seed=eval_seeds[0], keep_states=keep_states_seed is not None)
        return float(floor_pp), np.zeros_like(run.rmse), [run]
    runs = [run_config(store, model, colset, folds, rounds=rounds, seed=s,
                       keep_states=(keep_states_seed is not None and s == keep_states_seed)) for s in eval_seeds]
    tables = [r.rmse for r in runs]
    return seed_noise_eps(tables, floor_pp), seed_noise_cells(tables), runs


def run_at_seed(runs: list[RunResult], seed: int) -> RunResult | None:
    """Lấy run đã có ở đúng seed (tránh chạy lại khi selection_seed nằm trong eval_seeds)."""
    return next((r for r in runs if int(r.seed) == int(seed)), None)
