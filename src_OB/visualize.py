"""Post-hoc figures from saved artifacts only: per-origin forecast paths and E0-gain heatmaps.

No training, inference or GPU. Reads `predictions.parquet` of completed cells, `summary/per_fold_per_horizon.csv`
and the prepared raw mid timeline (actual prices between horizons). Origins follow a fixed clock rule on every
VAL day, never errors or predictions. Style follows RESEARCH_PLAN §7.3: actual black, E0 grey dashed, one fixed
colour and marker per model, diverging heatmaps on one shared scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .data import DAY, Data

MODELS = {  # family -> (label, colour, marker)
    "tfm_zero_shot": ("TimesFM zero-shot", "#4a3aa7", "P"),
    "tfm_lora": ("TimesFM LoRA", "#eb6834", "D"),
    "autots": ("AutoTS v2", "#e87ba4", "<"),
}
INK, MUTED, TRACE = "#0b0b0b", "#898781", "#bdbbb4"
CLOCK_HOURS = (4, 12, 20)  # first common origin inside each of these UTC hours of every VAL day
GAIN_LIMIT = 0.1  # heatmap colours clip here; every cell prints its exact value
MIN_DAY_ORIGINS = 1000  # a VAL day enters the day heatmap only with at least this many origins
HOUR = 3600 * 1_000_000


def utc(ts_us):
    return pd.Timestamp(int(ts_us), unit="us", tz="UTC")


def load_paths(output, folds, horizons):
    """One row per (fold, origin): origin mid, actual and every model's prediction at each horizon."""
    parts = []
    for fold in folds:
        for model in MODELS:
            for h in horizons:
                cell = output / fold.name / model / f"h{h}s"
                if not (cell / "completed.json").is_file():
                    raise FileNotFoundError(f"{cell} chưa completed; visualize chỉ đọc cell đã hoàn tất.")
                parts.append(pd.read_parquet(cell / "predictions.parquet").assign(fold=fold.name, model=model))
    long = pd.concat(parts, ignore_index=True)
    long["timestamp_us"] = long["timestamp_us"].astype(np.int64)
    long["horizon_seconds"] = long["horizon_seconds"].astype(int)
    key = ["fold", "timestamp_us"]
    predicted = long.pivot(index=key, columns=["model", "horizon_seconds"], values="predicted_price")
    actual = long.groupby(key + ["horizon_seconds"])["actual_price"].agg(["min", "max"])
    origin = long.groupby(key)["origin_price"].agg(["min", "max"])
    # Common origin mask: every model and horizon must score the same origins with the same actual and E0.
    if (predicted.isna().to_numpy().any() or not np.array_equal(actual["min"], actual["max"])
            or not np.array_equal(origin["min"], origin["max"])):
        raise ValueError("Origin/actual/E0 khác nhau giữa các family; không vẽ trên tập origin không chung.")
    table = pd.DataFrame({"origin": origin["min"]})
    by_h = actual["min"].unstack("horizon_seconds")
    for h in horizons:
        table[f"actual_{h}"] = by_h[h]
    for model in MODELS:
        for h in horizons:
            table[f"{model}_{h}"] = predicted[(model, h)]
    return table.reset_index()


def pick_origins(table, folds):
    """First common origin inside each clock hour of every UTC day that lies in VAL (fixed rule, not by error)."""
    picks = []
    for fold in folds:
        ts = np.sort(table.loc[table["fold"] == fold.name, "timestamp_us"].to_numpy())
        first_day = -(-fold.val_start // DAY) * DAY
        for day in range(first_day, fold.val_end, DAY):
            for hour in CLOCK_HOURS:
                start = day + hour * HOUR
                i = int(np.searchsorted(ts, start))
                if i < len(ts) and ts[i] < start + HOUR:
                    picks.append((fold.name, int(ts[i]), hour))
    return picks


def path_figure(fold, t, row, data, horizons, path):
    import matplotlib.pyplot as plt

    p0, unit = float(row["origin"]), horizons[0] * 1_000_000
    # Raw mid states from the origin to t + max horizon: the actual path between the scored points.
    lo = int(np.searchsorted(data.raw_ts, t, side="right")) - 1
    hi = int(np.searchsorted(data.raw_ts, t + horizons[-1] * 1_000_000, side="right"))
    raw_t = np.asarray(data.raw_ts[lo:hi], np.int64)
    raw_mid = np.asarray(data.raw_mid[lo:hi], np.float64)
    steps = np.r_[0, np.asarray(horizons) / horizons[0]]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.step(np.r_[np.maximum(raw_t - t, 0) / unit, steps[-1]], np.r_[raw_mid, raw_mid[-1]] - p0, where="post",
            color=TRACE, lw=1.0, label="mid thô giữa các mốc (mỗi snapshot)")
    ax.axhline(0, color=MUTED, ls="--", lw=1.3, label="E0 (giữ giá tại t)")
    for model, (label, colour, marker) in MODELS.items():
        predicted = np.r_[0, [row[f"{model}_{h}"] - p0 for h in horizons]]
        errors = " / ".join(f"{abs(row[f'{model}_{h}'] - row[f'actual_{h}']):.1f}" for h in horizons)
        ax.plot(steps, predicted, color=colour, marker=marker, lw=1.8, ms=7, label=f"{label}  |lỗi| {errors}")
    actual = np.r_[0, [row[f"actual_{h}"] - p0 for h in horizons]]
    errors = " / ".join(f"{abs(p0 - row[f'actual_{h}']):.1f}" for h in horizons)
    ax.plot(steps, actual, color=INK, marker="o", lw=2.2, ms=6, zorder=5, label=f"actual (mid as-of t+h)  |E0 lỗi| {errors}")
    ax.set_xticks(steps, ["t"] + [f"t+{k}\n(+{h} s)" for k, h in enumerate(horizons, 1)])
    ax.set_xlim(-0.08, steps[-1] + 0.08)
    ax.set_ylabel("Thay đổi giá so với mid tại t (USDT)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(f"{fold} · {utc(t):%Y-%m-%d %H:%M:%S} UTC · mid tại t = {p0:,.2f} USDT", fontsize=11)
    ax.set_title("Mỗi horizon là một model Direct riêng; |lỗi| tuyệt đối USDT theo t+1 / t+2 / t+3",
                 fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def day_gains(table, horizons):
    date = pd.to_datetime(table["timestamp_us"], unit="us", utc=True).dt.strftime("%Y-%m-%d")
    rows = []
    for (fold, day), group in table.groupby([table["fold"], date], sort=True):
        row = {"fold": fold, "day": day, "n": len(group)}
        for h in horizons:
            e0 = np.sqrt(np.mean((group["origin"] - group[f"actual_{h}"]) ** 2))
            for model in MODELS:
                rmse = np.sqrt(np.mean((group[f"{model}_{h}"] - group[f"actual_{h}"]) ** 2))
                row[f"{model}_{h}"] = 1 - rmse / e0
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap_figure(panels, rows, columns, title, note, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 1.6, 0.45 * len(rows) + 2.0), sharey=True)
    norm = Normalize(-GAIN_LIMIT, GAIN_LIMIT)
    for ax, (label, values) in zip(axes, panels.items()):
        image = ax.imshow(np.clip(values, -GAIN_LIMIT, GAIN_LIMIT), cmap="RdBu", norm=norm, aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) >= 0.6 * GAIN_LIMIT else INK)
        ax.set_xticks(range(len(columns)), columns)
        ax.set_title(label, fontsize=10)
    axes[0].set_yticks(range(len(rows)), rows)
    bar = fig.colorbar(image, ax=list(axes), shrink=0.85, pad=0.02)
    bar.set_label(f"rmse_gain_vs_e0 (màu cắt ở ±{GAIN_LIMIT:g})")
    fig.suptitle(title, fontsize=11)
    fig.text(0.01, -0.02, note, fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def visualize(cfg):
    import matplotlib
    matplotlib.use("Agg")

    output = Path(cfg["output_dir"])
    out = output / "figures"
    (out / "paths").mkdir(parents=True, exist_ok=True)
    horizons = list(cfg["horizons_seconds"])
    data = Data(cfg)
    folds = list(data.folds())
    table = load_paths(output, folds, horizons)
    indexed = table.set_index(["fold", "timestamp_us"])
    records = []
    for fold, t, hour in pick_origins(table, folds):
        row = indexed.loc[(fold, t)]
        name = f"path_{fold}_{utc(t):%Y%m%d_%H%M%S}.png"
        path_figure(fold, t, row, data, horizons, out / "paths" / name)
        record = {"file": f"paths/{name}", "fold": fold, "clock_hour_utc": hour, "origin_utc": utc(t).isoformat(),
                  "origin_mid": row["origin"]}
        for h in horizons:
            record[f"actual_{h}"] = row[f"actual_{h}"]
            for model in MODELS:
                record[f"{model}_{h}"] = row[f"{model}_{h}"]
        records.append(record)
    origins = pd.DataFrame(records)
    origins.to_csv(out / "origins.csv", index=False)

    columns = [f"h{h}" for h in horizons]
    summary = pd.read_csv(output / "summary" / "per_fold_per_horizon.csv")
    names = [f.name for f in folds]
    panels = {label: summary[summary["model"] == model].pivot(index="fold", columns="horizon_seconds",
                                                             values="rmse_gain_vs_e0").loc[names, horizons].to_numpy()
              for model, (label, _, _) in MODELS.items()}
    heatmap_figure(panels, names, columns, "Gain RMSE so với E0 theo fold × horizon",
                   "Nguồn: summary/per_fold_per_horizon.csv. Xanh: tốt hơn E0, đỏ: kém hơn E0.",
                   out / "heatmap_fold_horizon.png")
    days = day_gains(table, horizons)
    days.to_csv(out / "day_gains.csv", index=False)
    kept = days[days["n"] >= MIN_DAY_ORIGINS]
    panels = {label: kept[[f"{model}_{h}" for h in horizons]].to_numpy() for model, (label, _, _) in MODELS.items()}
    heatmap_figure(panels, [f"{r.day[5:]} ({r.fold}, n={r.n:,})" for r in kept.itertuples()], columns,
                   "Gain RMSE so với E0 theo ngày VAL (UTC) × horizon",
                   f"Tính từ predictions.parquet trên origin chung; bỏ ngày có < {MIN_DAY_ORIGINS} origin. "
                   "Xanh: tốt hơn E0, đỏ: kém hơn E0.", out / "heatmap_day_horizon.png")

    codes = {}
    for fold in folds:
        for model in MODELS:
            for h in horizons:
                run = json.loads((output / fold.name / model / f"h{h}s" / "run.json").read_text())
                codes.setdefault(MODELS[model][0], set()).add(run["train_code"]["code_commit"][:7])
    dropped = days[days["n"] < MIN_DAY_ORIGINS]
    shown = output.resolve().relative_to(ROOT) if output.resolve().is_relative_to(ROOT) else output
    lines = [
        "# Figures — TimesFM và AutoTS (hậu kỳ, chỉ từ artifact đã lưu)", "",
        f"Sinh bởi `python -m src_OB visualize --config <config>` từ `{shown}`. Không train, không inference, không GPU:",
        "chỉ đọc `predictions.parquet` của cell completed, `summary/per_fold_per_horizon.csv` và timeline mid thô đã prepare.", "",
        f"- Model: {', '.join(f'{label} (train code {', '.join(sorted(codes[label]))})' for label, _, _ in MODELS.values())}.",
        f"- Origin: ở mỗi ngày UTC nằm trong VAL, lấy origin chung đầu tiên trong các giờ {', '.join(f'{h:02d}:00' for h in CLOCK_HOURS)} UTC",
        "  (quy tắc cố định, không chọn theo lỗi hay dự báo). Mọi family và horizon chấm cùng origin, cùng actual/E0 (đã kiểm).",
        f"- Mỗi ảnh path: trục x = t → t+1 → t+2 → t+3 (h = {', '.join(f'{h} s' for h in horizons)}); trục y = thay đổi giá so với",
        "  mid tại t. Actual đen (mid as-of t+h, đúng giá dùng để chấm metric), E0 xám nét đứt, đường xám nhạt là mid thô",
        "  giữa các mốc; mỗi model một màu/marker cố định, nhãn ghi |lỗi| tuyệt đối (USDT) theo từng horizon.",
        f"- Heatmap: `heatmap_fold_horizon.png` (5 fold × {len(horizons)} h, từ summary) và `heatmap_day_horizon.png`",
        f"  ({len(days) - len(dropped)} ngày VAL × {len(horizons)} h, tính lại từ predictions, dữ liệu trong `day_gains.csv`). Thang màu",
        f"  diverging chung, cắt ở ±{GAIN_LIMIT:g}; số trong ô là giá trị thật.",
    ]
    if len(dropped):
        lines.append(f"- Ngày bị bỏ khỏi heatmap theo ngày (< {MIN_DAY_ORIGINS} origin): "
                     + ", ".join(f"{r.day} ({r.fold}, n={r.n})" for r in dropped.itertuples()) + ".")
    lines += ["", f"## Ảnh path ({len(origins)})", "",
              "| file | fold | origin (UTC) | mid tại t | Δ actual t+1 / t+2 / t+3 (USDT) |", "|---|---|---|---|---|"]
    for r in records:
        delta = " / ".join(f"{r[f'actual_{h}'] - r['origin_mid']:+.2f}" for h in horizons)
        lines.append(f"| `{r['file']}` | {r['fold']} | {r['origin_utc'][:19]} | {r['origin_mid']:,.2f} | {delta} |")
    (out / "index.md").write_text("\n".join(lines) + "\n")
    print(f"figures: {len(origins)} path images, 2 heatmaps -> {out}", flush=True)
