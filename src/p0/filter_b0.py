"""§1.4 Lọc 306 feature B0 → B0*: PI (xáo trong VAL) + SA (LightGBM 1 cột) + MI (FIT, null xáo trộn) → cờ ≥ 2/3 horizon
→ R1–R4 → 4 run kiểm chứng (15fixed_306) so với B0-306 → B0* = bộ không tệ hơn (≥ −ε) có MedianGain cao nhất (hòa → nhỏ hơn).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .harness import ColSet, RunResult, Store, run_config
from .metrics import cell_metrics, gain_pp, summarize
from .models import TabularModel
from .split import Fold


def permutation_importance(store: Store, run: RunResult, col_positions: list[int], repeats: int = 3, seed: int = 8586) -> np.ndarray:
    """ΔRMSE giá (USD) khi xáo cột j trong VAL (mỗi fold, mỗi horizon), trung bình `repeats` lần xáo. Trả (n_cols, F, 3).

    Tabular: giá trị cột j của mỗi origin bị thay bằng giá trị của origin khác trong VAL.
    Sequence (LSTM, `SeqBatch`): CÙNG logic đó ở mức mẫu — toàn bộ cửa sổ 512 phút của KÊNH j được lấy từ origin khác
    trong VAL (`perm`), không đụng vào các kênh khác, không đổi kiến trúc/context/training.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((len(col_positions), len(run.states), 3))
    for fi, st in enumerate(run.states):
        c_t, c_future, rv = store.targets(st.idx_val)
        base = cell_metrics(c_t, c_future, st.yhat)["rmse"]
        is_seq = hasattr(st.X_val, "with_perm")  # SeqBatch (LSTM) hoặc SeriesBatch (TimesFM covariate)
        X = st.X_val if is_seq else np.asarray(st.X_val)
        for cj, j in enumerate(col_positions):
            acc = np.zeros(3)
            for _ in range(repeats):
                if is_seq:
                    Xp = X.with_perm({int(j): rng.permutation(X.idx)})
                else:
                    Xp = X.copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                z = st.result.predict_z(Xp)
                yhat = np.asarray(z, np.float32) if st.result.is_logret else st.transform.decode(z, rv)
                acc += cell_metrics(c_t, c_future, yhat)["rmse"] - base
            out[cj, fi] = acc / repeats
    return out


def median_over_folds(x: np.ndarray) -> np.ndarray:
    """(n, F, 3) → (n, 3) median qua fold."""
    return np.median(x, axis=1)


def flag_2of3(score: np.ndarray) -> np.ndarray:
    """Cờ + khi điểm số > 0 ở ít nhất 2 trong 3 horizon (luật rev 8)."""
    return (np.asarray(score) > 0).sum(axis=1) >= 2


def standalone_gain(store: Store, model: TabularModel, folds: list[Fold], col_names: list[str], seed: int, base_rmse: np.ndarray | None,
                    progress=None) -> tuple[np.ndarray, np.ndarray]:
    """SA: LightGBM gốc chỉ trên [j] (ES), Gain trên giá vs E0 và vs base (B0-306) — (n, 3) median qua fold mỗi cái."""
    n = len(col_names)
    g_e0 = np.zeros((n, 3)); g_b0 = np.full((n, 3), np.nan)
    for k, name in enumerate(col_names):
        cs = ColSet((name,),) if name in store.b0_names else ColSet((), (name,))
        run = run_config(store, model, cs, folds, rounds=None, seed=seed, keep_states=False)
        g_e0[k] = np.median(gain_pp(run.rmse, run.e0), axis=0)
        if base_rmse is not None:
            g_b0[k] = np.median(gain_pp(run.rmse, base_rmse), axis=0)
        if progress:
            progress(k + 1, n, name)
    return g_e0, g_b0


def mutual_info(store: Store, folds: list[Fold], colset: ColSet, seed: int = 8586, n_neighbors: int = 3) -> np.ndarray:
    """MI(X_j, z_h) trên FIT của từng fold trừ MI với z xáo trộn; (n_cols, 3) median qua fold. Không phải training."""
    from sklearn.feature_selection import mutual_info_regression

    from .transform import TargetTransform

    n = len(colset.names)
    per_fold = np.zeros((len(folds), n, 3))
    rng = np.random.default_rng(seed)
    for fi, fold in enumerate(folds):
        idx = fold.fit.origins(store.ts, store.eligible)
        X = np.nan_to_num(store.matrix(idx, colset), nan=0.0)
        tr = TargetTransform.fit(store.fd.target, store.fd.rv60, idx)
        z = tr.encode(store.fd.target, store.fd.rv60, idx)
        for h in range(3):
            mi = mutual_info_regression(X, z[:, h], n_neighbors=n_neighbors, random_state=seed)
            mi_null = mutual_info_regression(X, rng.permutation(z[:, h]), n_neighbors=n_neighbors, random_state=seed)
            per_fold[fi, :, h] = mi - mi_null
    return np.median(per_fold, axis=0)


@dataclass
class FilterTable:
    names: list[str]
    pi: np.ndarray  # (n, 3)
    sa_e0: np.ndarray
    sa_b0: np.ndarray
    mi: np.ndarray

    def flags(self) -> dict[str, np.ndarray]:
        return {"PI": flag_2of3(self.pi), "SA": flag_2of3(self.sa_e0), "MI": flag_2of3(self.mi)}

    def sets(self) -> dict[str, tuple[str, ...]]:
        f = self.flags()
        pi, sa, mi = f["PI"], f["SA"], f["MI"]
        rules = {"R1": pi | sa | mi, "R2": pi | (sa & mi), "R3": pi, "R4": sa}
        return {k: tuple(n for n, keep in zip(self.names, v) if keep) for k, v in rules.items()}

    def to_frame(self) -> pd.DataFrame:
        f, s = self.flags(), self.sets()
        rows = []
        for i, n in enumerate(self.names):
            base, lag = (n.split(":")[-1], n.split(":")[1] if n.count(":") == 2 else "t")
            rows.append({
                "col": n, "base": base, "lag": lag,
                **{f"PI_h{h + 1}": self.pi[i, h] for h in range(3)},
                **{f"SA_gain_e0_h{h + 1}": self.sa_e0[i, h] for h in range(3)},
                **{f"SA_gain_b0306_h{h + 1}": self.sa_b0[i, h] for h in range(3)},
                **{f"MI_minus_null_h{h + 1}": self.mi[i, h] for h in range(3)},
                "PI_plus": bool(f["PI"][i]), "SA_plus": bool(f["SA"][i]), "MI_plus": bool(f["MI"][i]),
                **{f"keep_{k}": (n in set(v)) for k, v in s.items()},
            })
        return pd.DataFrame(rows)


def verify_sets(store: Store, model: TabularModel, sets: dict[str, tuple[str, ...]], folds: list[Fold], rounds, base_rmse: np.ndarray,
                eps: float, seed: int = 8586) -> tuple[pd.DataFrame, str, dict[str, RunResult]]:
    """4 run kiểm chứng R1–R4 với 15fixed_306 → chọn B0*."""
    rows, runs = [], {}
    for k, cols in sets.items():
        if len(cols) == 0:
            rows.append({"set": k, "n_cols": 0, "MedianGain": np.nan, "WinRate": np.nan, "P10Gain": np.nan, "WorstGain": np.nan, "eligible": False})
            continue
        run = run_config(store, model, ColSet(tuple(cols)), folds, rounds=rounds, seed=seed, keep_states=False)
        runs[k] = run
        s = summarize(run.gain_vs(base_rmse))
        rows.append({"set": k, "n_cols": len(cols), **{kk: s[kk] for kk in ("MedianGain", "WinRate", "P10Gain", "WorstGain")},
                     "eligible": bool(s["MedianGain"] >= -eps)})
    df = pd.DataFrame(rows)
    elig = df[df["eligible"]]
    if len(elig) == 0:
        chosen = "B0-306"
    else:
        best = float(elig["MedianGain"].max())
        near = elig[elig["MedianGain"] >= best - eps]
        chosen = str(near.sort_values("n_cols").iloc[0]["set"])
    df["chosen"] = df["set"] == chosen
    return df, chosen, runs
