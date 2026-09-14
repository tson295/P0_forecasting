#!/usr/bin/env python3
"""Render the completed TimesFM/AutoTS phase from saved VAL artifacts only.

This post-processing command deliberately imports no model or training pipeline.
It reads the canonical 1-minute close timeline, final-representative predictions,
and saved metrics, then writes deterministic forecast-path overlays and heatmaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


HORIZONS = (1, 2, 3)
MODEL_ORDER = ("tfm", "autots")
MODEL_LABEL = {"tfm": "TimesFM-final", "autots": "AutoTS-final"}
MODEL_STYLE = {
    "tfm": {"color": "#2a78d6", "marker": "^"},
    "autots": {"color": "#e34948", "marker": "o"},
}


@dataclass(frozen=True)
class FoldPredictions:
    name: str
    idx: np.ndarray
    yhat: np.ndarray


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def script_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_timeline(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    ts_column = "timestamp" if "timestamp" in frame.columns else "ts" if "ts" in frame.columns else None
    close_column = "close" if "close" in frame.columns else "Close" if "Close" in frame.columns else None
    if ts_column is None or close_column is None:
        raise ValueError(f"{csv_path}: expected timestamp/ts and close/Close columns")
    ts = pd.to_numeric(frame[ts_column], errors="raise").to_numpy(dtype=np.int64)
    close = pd.to_numeric(frame[close_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(f"{csv_path}: close must be finite and positive")
    if len(ts) < 4 or not np.all(np.diff(ts) > 0):
        raise ValueError(f"{csv_path}: timeline must already be strictly increasing; refusing to sort/reindex")
    return ts, close


def load_predictions(exp_dir: Path, model: str, seed_index: int, folds: list[str]) -> list[FoldPredictions]:
    path = exp_dir / "wins" / f"{model}_seed{seed_index}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        expected = {f"idx_{i}" for i in range(len(folds))} | {f"yhat_{i}" for i in range(len(folds))}
        if set(data.files) != expected:
            raise ValueError(f"{path}: unexpected fold keys {data.files}")
        rows: list[FoldPredictions] = []
        for i, fold in enumerate(folds):
            idx = np.asarray(data[f"idx_{i}"], dtype=np.int64)
            yhat = np.asarray(data[f"yhat_{i}"], dtype=float)
            if idx.ndim != 1 or yhat.shape != (len(idx), len(HORIZONS)):
                raise ValueError(f"{path}: invalid {fold} shapes idx={idx.shape}, yhat={yhat.shape}")
            if len(idx) == 0 or not np.all(np.diff(idx) > 0) or not np.isfinite(yhat).all():
                raise ValueError(f"{path}: invalid values for {fold}")
            rows.append(FoldPredictions(fold, idx, yhat))
    return rows


def valid_origins(idx: np.ndarray, ts: np.ndarray) -> np.ndarray:
    ok = (idx >= 0) & (idx + max(HORIZONS) < len(ts))
    result = idx[ok]
    return result[np.all(np.stack([ts[result + h] == ts[result] + h * 60 for h in HORIZONS]), axis=0)]


def select_origins(common: np.ndarray, ts: np.ndarray, wanted: int) -> np.ndarray:
    """One origin nearest the midpoint of each equal time interval, without repeats."""
    if wanted <= 0 or not len(common):
        return np.empty(0, dtype=np.int64)
    count = min(wanted, len(common))
    origin_ts = ts[common]
    edges = np.linspace(float(origin_ts[0]), float(origin_ts[-1]), count + 1)
    picks: list[int] = []
    used: set[int] = set()
    for left, right in zip(edges[:-1], edges[1:]):
        candidates = common[(origin_ts >= left) & (origin_ts <= right)]
        if not len(candidates):
            continue
        midpoint = (left + right) / 2
        ordered = candidates[np.argsort(np.abs(ts[candidates] - midpoint), kind="stable")]
        chosen = next((int(value) for value in ordered if int(value) not in used), None)
        if chosen is not None:
            picks.append(chosen)
            used.add(chosen)
    return np.asarray(picks, dtype=np.int64)


def per_fold_counts(n_folds: int, total: int) -> list[int]:
    base, extra = divmod(total, n_folds)
    return [base + int(i < extra) for i in range(n_folds)]


def plot_path(out: Path, number: int, fold: str, origin: int, ts: np.ndarray, close: np.ndarray,
              tfm: np.ndarray, autots: np.ndarray, actual_seed: int) -> dict:
    base = close[origin]
    actual = np.r_[0.0, close[origin + np.asarray(HORIZONS)] - base]
    paths = {
        "tfm": np.r_[0.0, base * np.exp(tfm) - base],
        "autots": np.r_[0.0, base * np.exp(autots) - base],
    }
    x = np.arange(4)
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(x, actual, color="#111111", marker="o", linewidth=1.8, label="Actual")
    axis.plot(x, np.zeros(4), color="#777777", linestyle="--", linewidth=1.3, label="E0 (zero return)")
    for model in MODEL_ORDER:
        style = MODEL_STYLE[model]
        axis.plot(x, paths[model], color=style["color"], marker=style["marker"], linewidth=1.5,
                  label=MODEL_LABEL[model])
    timestamp = pd.Timestamp(ts[origin], unit="s", tz="UTC")
    axis.set_xticks(x, ["t", "+1 min", "+2 min", "+3 min"])
    axis.set_ylabel("Change from $C_t$ (USD)")
    axis.set_xlabel("Forecast path from one origin")
    axis.set_title(f"TimesFM-final vs AutoTS-final | {fold} | {timestamp:%Y-%m-%d %H:%M UTC} | C_t=${base:,.2f} | seed {actual_seed}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", frameon=False, ncol=2)
    filename = f"path_{number:03d}_{fold}_{timestamp:%Y%m%dT%H%M%SZ}.png"
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    return {
        "file": filename, "figure_type": "forecast_path_overlay", "fold": fold, "origin_idx": int(origin),
        "timestamp_utc": timestamp.isoformat(), "actual_seed": actual_seed,
        "selection_rule": "equal-time interval midpoint among common valid seed-0 origins",
        "units": "USD change from origin close", "source_metric": "saved predictions",
    }


def calculate_metrics(predictions: dict[str, list[list[FoldPredictions]]], wins: dict[str, dict],
                      ts: np.ndarray, close: np.ndarray, eval_seeds: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        saved_e0 = np.asarray(wins[model]["e0"], dtype=float)
        for seed_index, folds in enumerate(predictions[model]):
            for fold_index, fold_data in enumerate(folds):
                idx = valid_origins(fold_data.idx, ts)
                if len(idx) != len(fold_data.idx):
                    raise ValueError(f"{model}/{fold_data.name}: prediction origins cross a timestamp gap")
                for h_index, horizon in enumerate(HORIZONS):
                    actual = close[idx + horizon]
                    pred = close[idx] * np.exp(fold_data.yhat[:, h_index])
                    error = pred - actual
                    rmse = float(np.sqrt(np.mean(error ** 2)))
                    mae = float(np.mean(np.abs(error)))
                    centered = float(np.sum((actual - np.mean(actual)) ** 2))
                    r2 = float(1 - np.sum(error ** 2) / centered) if centered else np.nan
                    e0 = float(saved_e0[fold_index, h_index])
                    rows.append({
                        "record_type": "seed", "model": model, "fold": fold_data.name,
                        "horizon_minutes": horizon, "seed": int(eval_seeds[seed_index]), "n_origins": len(idx),
                        "rmse_usd": rmse, "mae_usd": mae, "r2": r2, "rmse_e0_saved_usd": e0,
                        "rmse_gain_vs_e0": 1 - rmse / e0 if e0 else np.nan,
                        "r2_os_vs_e0": 1 - (rmse ** 2) / (e0 ** 2) if e0 else np.nan,
                        "aggregation": "seed-specific metric from saved prediction",
                    })
    seed_rows = pd.DataFrame(rows)
    summary = seed_rows.groupby(["model", "fold", "horizon_minutes"], as_index=False).agg(
        n_origins=("n_origins", "first"), rmse_usd=("rmse_usd", "mean"), mae_usd=("mae_usd", "mean"),
        r2=("r2", "mean"), rmse_e0_saved_usd=("rmse_e0_saved_usd", "first"),
        rmse_gain_vs_e0=("rmse_gain_vs_e0", "mean"), r2_os_vs_e0=("r2_os_vs_e0", "mean"),
    )
    summary.insert(0, "record_type", "mean_over_eval_seeds")
    summary.insert(4, "seed", "all")
    summary["aggregation"] = "mean over seed-specific metrics; E0 reused from wins artifact"
    return pd.concat([seed_rows, summary], ignore_index=True)


def plot_heatmap(out: Path, metric: str, title: str, source: pd.DataFrame, folds: list[str],
                 diverging: bool, filename: str) -> dict:
    values = []
    for model in MODEL_ORDER:
        table = source[source["model"] == model].pivot(index="fold", columns="horizon_minutes", values=metric)
        table = table.reindex(index=folds, columns=HORIZONS)
        values.append(table.to_numpy(dtype=float))
    finite = np.concatenate([array[np.isfinite(array)] for array in values])
    if not len(finite):
        raise ValueError(f"No finite values for {metric}")
    if diverging:
        extent = max(abs(float(finite.min())), abs(float(finite.max())), 1e-12)
        norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
        cmap = "RdBu_r"
    else:
        low, high = float(finite.min()), float(finite.max())
        norm = Normalize(vmin=low - 0.02 * (high - low or 1), vmax=high + 0.02 * (high - low or 1))
        cmap = "viridis"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    image = None
    for axis, model, matrix in zip(axes, MODEL_ORDER, values):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        axis.set_title(MODEL_LABEL[model])
        axis.set_xticks(range(3), ["h1", "h2", "h3"])
        axis.set_yticks(range(len(folds)), folds)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if np.isfinite(value):
                    axis.text(col, row, f"{value:.4f}", ha="center", va="center", fontsize=8,
                              color="white" if norm(value) < 0.45 else "black")
    axes[0].set_ylabel("VAL fold")
    fig.suptitle(title)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.9)
    colorbar.set_label(metric)
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)
    return {
        "file": filename, "figure_type": "heatmap", "fold": "all", "origin_idx": "",
        "timestamp_utc": "", "actual_seed": "all", "selection_rule": "all saved fold/horizon cells",
        "units": "USD" if metric.endswith("usd") else "ratio", "source_metric": metric,
    }


def run(config_path: Path, out: Path | None, n_paths: int) -> Path:
    root = project_root()
    config = read_json(config_path)
    exp_dir = root / config["experiments_dir"]
    output = (root / out if out and not out.is_absolute() else out) if out else exp_dir / "visualize"
    assert output is not None
    output.mkdir(parents=True, exist_ok=True)
    contract = read_json(exp_dir / "phase_config.json")
    progress = read_json(exp_dir / "phase_progress.json")
    if progress.get("status") != "completed":
        raise ValueError("Visualization requires completed phase_progress.json")
    if contract.get("config_hash") != read_json(exp_dir / "phase_config.json").get("config_hash"):
        raise ValueError("Invalid phase contract")
    timeline_path = root / config["hf_csv"]
    ts, close = load_timeline(timeline_path)
    wins = {model: read_json(exp_dir / "wins" / f"{model}.json") for model in MODEL_ORDER}
    folds = wins["tfm"].get("folds")
    if not isinstance(folds, list) or not folds or wins["autots"].get("folds") != folds:
        raise ValueError("Final representatives have incompatible fold metadata")
    eval_seeds = wins["tfm"].get("eval_seeds")
    if not isinstance(eval_seeds, list) or len(eval_seeds) != 3 or wins["autots"].get("eval_seeds") != eval_seeds:
        raise ValueError("Final representatives have incompatible eval seeds")
    predictions = {model: [load_predictions(exp_dir, model, seed_i, folds) for seed_i in range(len(eval_seeds))]
                   for model in MODEL_ORDER}
    metrics = calculate_metrics(predictions, wins, ts, close, eval_seeds)
    metrics.to_csv(output / "visualization_metrics.csv", index=False)

    manifest: list[dict] = []
    values: list[dict] = []
    path_number = 1
    for fold_i, (fold, count) in enumerate(zip(folds, per_fold_counts(len(folds), n_paths))):
        tfm_fold = predictions["tfm"][0][fold_i]
        autots_fold = predictions["autots"][0][fold_i]
        common = np.intersect1d(valid_origins(tfm_fold.idx, ts), valid_origins(autots_fold.idx, ts), assume_unique=True)
        picks = select_origins(common, ts, count)
        tfm_pos = {int(value): i for i, value in enumerate(tfm_fold.idx)}
        autots_pos = {int(value): i for i, value in enumerate(autots_fold.idx)}
        for origin in picks:
            tfm_yhat = tfm_fold.yhat[tfm_pos[int(origin)]]
            autots_yhat = autots_fold.yhat[autots_pos[int(origin)]]
            manifest.append(plot_path(output, path_number, fold, int(origin), ts, close, tfm_yhat, autots_yhat,
                                      int(eval_seeds[0])))
            base = close[origin]
            for h_i, horizon in enumerate(HORIZONS):
                values.append({
                    "path_number": path_number, "fold": fold, "origin_idx": int(origin),
                    "timestamp_utc": pd.Timestamp(ts[origin], unit="s", tz="UTC").isoformat(),
                    "horizon_minutes": horizon, "actual_seed": int(eval_seeds[0]), "origin_close_usd": base,
                    "actual_price_usd": close[origin + horizon], "e0_price_usd": base,
                    "tfm_price_usd": base * np.exp(tfm_yhat[h_i]),
                    "autots_price_usd": base * np.exp(autots_yhat[h_i]),
                    "actual_delta_usd": close[origin + horizon] - base,
                    "e0_delta_usd": 0.0, "tfm_delta_usd": base * np.exp(tfm_yhat[h_i]) - base,
                    "autots_delta_usd": base * np.exp(autots_yhat[h_i]) - base,
                })
            path_number += 1
    pd.DataFrame(values).to_csv(output / "plot_values.csv", index=False)

    saved_cells = pd.read_csv(exp_dir / "tfm_autots_per_fold_horizon.csv")
    saved_cells["fold"] = saved_cells["fold"].map(lambda value: folds[int(value) - 1])
    mean_metrics = metrics[metrics["record_type"] == "mean_over_eval_seeds"]
    manifest += [
        plot_heatmap(output, "rmse_gain_vs_e0", "RMSE gain vs E0 (saved phase summary)", saved_cells, folds, True,
                     "heatmap_rmse_gain_vs_e0.png"),
        plot_heatmap(output, "r2_os_vs_e0", "R² OS vs E0 (saved phase summary)", saved_cells, folds, True,
                     "heatmap_r2_os_vs_e0.png"),
        plot_heatmap(output, "rmse_usd", "RMSE from saved predictions", mean_metrics, folds, False,
                     "heatmap_rmse_usd.png"),
        plot_heatmap(output, "mae_usd", "MAE from saved predictions", mean_metrics, folds, False,
                     "heatmap_mae_usd.png"),
        plot_heatmap(output, "r2", "Standard R² from saved predictions", mean_metrics, folds, True,
                     "heatmap_r2.png"),
    ]
    for row in manifest:
        row.update({
            "run_config_hash": contract.get("config_hash"), "phase_code_hash": contract.get("code_hash"),
            "visualizer_code_hash": script_hash(), "git_revision": git_revision(root),
            "timeline": str(timeline_path.relative_to(root)), "prediction_sources": "wins/tfm_seed*.npz,wins/autots_seed*.npz",
        })
    pd.DataFrame(manifest).to_csv(output / "figure_manifest.csv", index=False)
    report = f"""# TimesFM / AutoTS visualization

- Output: `{output.relative_to(root)}`
- Run/config provenance: phase config hash `{contract.get('config_hash')}`, phase code hash `{contract.get('code_hash')}`, visualizer hash `{script_hash()}`.
- Inputs: `{timeline_path.relative_to(root)}`, final representative `wins/tfm*.json/.npz`, `wins/autots*.json/.npz`, and saved summary tables.
- Forecast paths: {path_number - 1} PNG overlays across {len(folds)} real VAL folds; seed `{eval_seeds[0]}` only, selected deterministically from shared valid origins without inspecting errors.
- Heatmaps: five PNGs. Gain/R² OS use the saved phase cells; RMSE/MAE/standard R² are calculated from the saved prediction arrays over all valid origins and then averaged over eval seeds.
- No model, checkpoint, feature store, training, refit, inference, test holdout, latency replay, smoke, probe, or benchmark was run. MAE is a post-hoc metric from saved predictions; latency remains unavailable.
"""
    (output / "VISUALIZE_REPORT.md").write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tfm_autots.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-paths", type=int, default=30)
    args = parser.parse_args()
    if args.n_paths < 1:
        raise ValueError("--n-paths must be positive")
    output = run(args.config, args.out, args.n_paths)
    print(output)


if __name__ == "__main__":
    main()
