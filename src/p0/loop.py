"""§2.1 add-one loop → F*_m; prune PI → F*_m^prune; confirmation 3 seed (mean RMSE từng ô) → win_m; §3 champion; ensemble."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features_ext import Candidate
from .filter_b0 import flag_2of3, median_over_folds, permutation_importance
from .harness import ColSet, RunResult, Store, run_config
from .metrics import decide, gain_pp, mean_rmse_over_seeds, summarize
from .models import TabularModel
from .split import Fold


@dataclass
class LoopResult:
    final: ColSet
    final_rmse: np.ndarray
    table: pd.DataFrame
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def add_one_loop(store: Store, model: TabularModel, base: ColSet, base_rmse: np.ndarray, candidates: list[Candidate], folds: list[Fold],
                 rounds, eps: float, seed: int, e0_rmse: np.ndarray, standalone_fn=None, on_row=None) -> LoopResult:
    """S := base; với từng candidate: thêm → train (số vòng cố định) → Gain vs S → KEEP nếu ≥ −ε, DROP nếu < −ε.
    on_row(row: dict, run: RunResult) được gọi sau mỗi candidate (log / save_run; có thể thêm exp_id vào row)."""
    S, S_rmse = base, base_rmse
    rows, kept, dropped = [], [], []
    for i, cand in enumerate(candidates, start=1):
        cs = S.with_ext(cand.columns)
        run = run_config(store, model, cs, folds, rounds=rounds, seed=seed, keep_states=False)
        g = run.gain_vs(S_rmse)
        s = summarize(g)
        decision = decide(s["MedianGain"], eps)
        row = {
            "order": i, "candidate": cand.name, "columns": "|".join(cand.columns), "group": cand.group,
            "MedianGain_vs_S": s["MedianGain"], "WinRate": s["WinRate"], "P10Gain": s["P10Gain"], "WorstGain": s["WorstGain"],
            "MedianGain_vs_base": float(np.median(run.gain_vs(base_rmse))), "MedianGain_vs_E0": float(np.median(run.gain_vs(e0_rmse))),
            "gain_standalone_E0": float(standalone_fn(cand)) if standalone_fn else np.nan,
            "decision": decision, "eps": eps, "size_S_after": len(cs.names) if decision == "KEEP" else len(S.names),
            "rmse_cells": json.dumps(np.round(run.rmse, 4).tolist()), "gain_cells_vs_S": json.dumps(np.round(g, 4).tolist()),
        }
        if decision == "KEEP":
            S, S_rmse = cs, run.rmse
            kept.append(cand.name)
        else:
            dropped.append(cand.name)
        if on_row:
            on_row(row, run)  # có thể thêm exp_id vào row trước khi ghi bảng
        rows.append(row)
    return LoopResult(S, S_rmse, pd.DataFrame(rows), kept, dropped)


def prune_pi(store: Store, model: TabularModel, colset: ColSet, folds: list[Fold], rounds, seed: int, repeats: int = 3) -> tuple[ColSet, pd.DataFrame]:
    """Prune PI (§2.1a): PI trên VAL cho các cột ext; bỏ cột không có cờ PI+ (PI ≤ 0 ở ≥ 2/3 horizon)."""
    if not colset.ext:
        return colset, pd.DataFrame(columns=["col", "PI_h1", "PI_h2", "PI_h3", "keep"])
    run = run_config(store, model, colset, folds, rounds=rounds, seed=seed, keep_states=True)
    kind = getattr(model, "input_kind", "tabular")
    if kind == "sequence":  # LSTM: input là kênh của fine_matrix, không phải cột của ma trận B0 + ext
        names = store.fine_names(colset)
        positions = [names.index(c) for c in colset.ext]
    elif kind == "series":  # covariate = ext (TimesFM) hoặc B0* + ext (AutoTS) → ext nằm ở cuối
        off = len(colset.b0) if getattr(model, "series_covariates", "ext") == "all" else 0
        positions = [off + i for i in range(len(colset.ext))]
    else:
        positions = [len(colset.b0) + i for i in range(len(colset.ext))]
    delta = median_over_folds(permutation_importance(store, run, positions, repeats=repeats, seed=seed))
    keep = flag_2of3(delta)
    df = pd.DataFrame({"col": list(colset.ext), "PI_h1": delta[:, 0], "PI_h2": delta[:, 1], "PI_h3": delta[:, 2], "keep": keep})
    pruned = ColSet(colset.b0, tuple(c for c, k in zip(colset.ext, keep) if k))
    return pruned, df


@dataclass
class Confirmed:
    colset: ColSet
    runs: list[RunResult]
    rmse_mean: np.ndarray  # RMSE̅ (F, 3): mean 3 seed từng ô
    e0: np.ndarray

    @property
    def best_iters(self) -> list[np.ndarray]:
        return [r.best_iters for r in self.runs]

    def preds_by_seed(self) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        return [r.preds() for r in self.runs]


def confirm(store: Store, model: TabularModel, colset: ColSet, folds: list[Fold], seeds, keep_states: bool = True) -> Confirmed:
    """Confirmation (§2.1b): 3 evaluation seed, ES bật → 3 bảng RMSE → RMSE̅ = mean từng ô.
    Model tất định (TimesFM zero-shot) chạy MỘT lần: 3 seed sẽ cho kết quả y hệt, RMSE̅ = chính bảng đó."""
    used = list(seeds)[:1] if not getattr(model, "seed_dependent", True) else list(seeds)
    runs = [run_config(store, model, colset, folds, rounds=None, seed=s, keep_states=keep_states) for s in used]
    return Confirmed(colset, runs, mean_rmse_over_seeds([r.rmse for r in runs]), runs[0].e0)


def decide_win(unpruned: Confirmed, pruned: Confirmed, eps: float) -> tuple[str, np.ndarray, dict]:
    """Gain_{f,h} = 1 − RMSE̅^prune / RMSE̅^unprune; MedianGain ≥ −ε → prune; thấp hơn → unpruned."""
    g = gain_pp(pruned.rmse_mean, unpruned.rmse_mean)
    s = summarize(g)
    return ("prune" if s["MedianGain"] >= -eps else "unprune"), g, s


def compare(win_rmse: np.ndarray, champ_rmse: np.ndarray, eps: float) -> tuple[bool, np.ndarray, dict]:
    """§3: Gain từng ô = 1 − RMSE̅_win / RMSE̅_champion; đổi champion khi MedianGain > +ε_champion."""
    g = gain_pp(win_rmse, champ_rmse)
    s = summarize(g)
    return bool(s["MedianGain"] > eps), g, s


# ----------------------------------------------------------------------------- ensemble (§3)
def ensemble_rmse(store: Store, members: dict[str, list[list[tuple[np.ndarray, np.ndarray]]]], folds: list[Fold],
                  weights: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """RMSE̅ của ensemble: với mỗi seed k và fold f, ŷ_ens = Σ_m w_m,h · ŷ_m (cùng origin), RMSE giá; mean qua seed.
    members: model → [seed][fold] (idx_val, yhat). weights: model → (3,) hoặc None = đều."""
    names = list(members)
    n_seed = min(len(members[m]) for m in names)
    F = len(folds)
    rmse = np.zeros((n_seed, F, 3))
    for k in range(n_seed):
        for f in range(F):
            idx = members[names[0]][k][f][0]
            acc = np.zeros((len(idx), 3))
            wsum = np.zeros(3)
            for m in names:
                idx_m, yhat_m = members[m][k][f]
                assert np.array_equal(idx_m, idx), "origin không khớp giữa các thành viên ensemble"
                w = np.ones(3) if weights is None else np.asarray(weights[m], float)
                acc += yhat_m * w[None, :]
                wsum += w
            yhat = acc / wsum[None, :]
            c_t, c_future, _ = store.targets(idx)
            from .metrics import cell_metrics

            rmse[k, f] = cell_metrics(c_t, c_future, yhat)["rmse"]
    return rmse.mean(axis=0)


def inverse_mse_weights(rmse_mean_by_model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """w_m,h ∝ 1 / mean_f(RMSE̅_m,f,h²)."""
    return {m: 1.0 / np.mean(np.asarray(t, float) ** 2, axis=0) for m, t in rmse_mean_by_model.items()}


# ----------------------------------------------------------------------------- champion state
def load_champion(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_champion(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
