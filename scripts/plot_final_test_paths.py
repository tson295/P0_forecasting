"""Post-hoc visualization của FINAL TEST (một lần, `python run.py final`) — KHÔNG train/inference, chỉ đọc artifact:
`experiments/<run>/final/index.json` + `final/<key>.npz` (prediction đã lưu) + data thật (để dựng lại giá thật/vol).

15 figure `test_paths_NN.png` dưới `reports/figures/final_test_paths/`, mỗi figure = lưới 2×3:
  hàng trên  = nhóm tree/ensemble (lgbm, xgb, cat, xgbrf, ensemble) — chỉ vẽ model NÀO thực sự có trong final/index.json;
  hàng dưới  = nhóm khác/tham chiếu (lstm, B0-306, B0*) — chỉ vẽ model NÀO thực sự có;
  3 cột      = 3 origin TEST khác nhau (một bộ low/medium/high theo rv60, tiêu chí chọn khác nhau mỗi figure/theme)
             → 15 figure × 3 origin = 45 origin, không trùng nhau giữa các figure.
x = t, t+1, t+2, t+3 (phút); y = P̂_(t+h) − C_t = C_t·(exp(ŷ_h) − 1) (USD, mọi đường xuất phát từ 0 tại t);
actual = đen; E0 = 0 (đường ngang nét đứt xám). Không đổi quyết định model/feature/champion — thuần phân tích hậu kỳ.

Chạy: python scripts/plot_final_test_paths.py --config configs/p0_ml_lstm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from p0.cli import load_store  # noqa: E402
from p0.config import HORIZONS  # noqa: E402

# ----------------------------------------------------------------------------- style (cố định model -> màu/marker, mọi figure)
# Bảng màu categorical đã kiểm CVD (dataviz skill, references/palette.md) — thứ tự CỐ ĐỊNH, không cycle theo model nào có mặt.
TOP_MODELS = ["lgbm", "xgb", "cat", "xgbrf", "ensemble"]      # hàng trên: tree/ensemble
BOTTOM_MODELS = ["lstm", "b0_306", "b0_star"]                  # hàng dưới: khác/tham chiếu (B0-306/B0* = tham chiếu B0/S0)
MODEL_LABEL = {"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "xgbrf": "XGB-RF", "ensemble": "Ensemble",
              "lstm": "LSTM", "b0_306": "B0-306 (ref)", "b0_star": "B0* (ref)"}
MODEL_STYLE = {  # (màu hex, marker) — 8 slot categorical palette đã validate; cố định theo model, không đổi giữa các figure
    "lgbm":     ("#2a78d6", "o"),
    "xgb":      ("#eb6834", "s"),
    "cat":      ("#1baf7a", "^"),
    "xgbrf":    ("#eda100", "D"),
    "ensemble": ("#e87ba4", "v"),
    "lstm":     ("#008300", "P"),
    "b0_306":   ("#4a3aa7", "X"),
    "b0_star":  ("#e34948", "*"),
}
ACTUAL_STYLE = ("#0b0b0b", None)
GRID_KW = dict(color="#c9c8c2", linewidth=0.6, alpha=0.7)
X = np.array([0, 1, 2, 3])


def _load_final(exp: Path) -> tuple[dict, dict]:
    """index.json + {key: (idx, yhat (n,3))} — chỉ đọc, không suy luận lại."""
    idx_path = exp / "final" / "index.json"
    if not idx_path.exists():
        sys.exit(f"Thiếu {idx_path} — chạy `python run.py final` (một lần) trước khi visualize.")
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    preds = {}
    for key in index["keys"]:
        p = exp / "final" / f"{key}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        preds[key] = (z["idx_0"], z["yhat_0"])
    return index, preds


def _series_available(preds: dict, wanted: list[str]) -> list[str]:
    """Chỉ trả model THỰC SỰ có prediction FINAL TEST — không bịa TimesFM/AutoTS hay bất kỳ key nào không tồn tại."""
    return [m for m in wanted if m in preds]


def build_frame(store, index: dict, preds: dict) -> pd.DataFrame:
    """Một hàng / origin TEST: actual C_(t+h)-C_t (USD), vol (rv60, bp), và pred_delta_h mỗi model có sẵn."""
    any_key = next(iter(preds))
    idx = preds[any_key][0]
    for k, (idx_k, _) in preds.items():
        if not np.array_equal(idx_k, idx):
            raise ValueError(f"final/{k}.npz: origin TEST không khớp final/{any_key}.npz — artifact không nhất quán")
    c_t, c_future, rv60 = store.targets(idx)
    ts = store.ts[idx]
    df = pd.DataFrame({"origin_idx": idx, "ts": ts, "c_t": c_t, "rv60_bp": rv60 * 1e4})
    for h in HORIZONS:
        df[f"actual_h{h}"] = c_future[:, h - 1] - c_t
    for key, (_, yhat) in preds.items():
        p_price = c_t[:, None] * np.exp(yhat)  # P̂_h = C_t · exp(ŷ_h) — đúng công thức metric trên giá của plan
        for h in HORIZONS:
            df[f"pred_{key}_h{h}"] = p_price[:, h - 1] - c_t
    return df


def _add_metrics(df: pd.DataFrame, all_models: list[str]) -> pd.DataFrame:
    err = np.stack([np.abs(df[f"pred_{m}_h3"] - df["actual_h3"]) for m in all_models], axis=1)
    df["avg_abs_err_h3"] = err.mean(axis=1)
    pred_h3 = np.stack([df[f"pred_{m}_h3"] for m in all_models], axis=1)
    df["disagreement_h3"] = pred_h3.std(axis=1)
    if "xgb" in all_models:  # xgb = champion (VAL đã chọn, không chọn lại ở đây)
        df["champion_err_h3"] = np.abs(df["pred_xgb_h3"] - df["actual_h3"])
        df["champion_gain_h3"] = np.abs(df["actual_h3"]) - df["champion_err_h3"]  # >0: champion sát thực tế hơn "đoán 0" (E0)
    tree_models = [m for m in ("lgbm", "xgb", "cat", "xgbrf") if m in all_models]
    if tree_models and "lstm" in all_models:
        df["lstm_vs_tree_div_h3"] = df["pred_lstm_h3"] - np.mean([df[f"pred_{m}_h3"] for m in tree_models], axis=0)
    feat_models = [m for m in ("lgbm", "xgb", "cat", "xgbrf") if m in all_models]
    ref_models = [m for m in ("b0_306", "b0_star") if m in all_models]
    if feat_models and ref_models:
        df["feat_vs_ref_div_h3"] = np.mean([df[f"pred_{m}_h3"] for m in feat_models], axis=0) - \
            np.mean([df[f"pred_{m}_h3"] for m in ref_models], axis=0)
    df["reversal_h3"] = np.sign(df["actual_h1"]) != np.sign(df["actual_h3"])
    df["monotonic_h3"] = (np.sign(df["actual_h1"]) == np.sign(df["actual_h2"])) & (np.sign(df["actual_h2"]) == np.sign(df["actual_h3"])) & \
        (df["actual_h1"].abs() <= df["actual_h2"].abs()) & (df["actual_h2"].abs() <= df["actual_h3"].abs())
    return df


REGIME_LABEL = {0: "Low", 1: "Medium", 2: "High"}


def _tertile(df: pd.DataFrame) -> pd.Series:
    q1, q2 = df["rv60_bp"].quantile([1 / 3, 2 / 3])
    return pd.cut(df["rv60_bp"], bins=[-np.inf, q1, q2, np.inf], labels=[0, 1, 2]).astype(int)


# ----------------------------------------------------------------------------- 15 theme: mỗi theme chọn 1 origin / tertile
def pick_themes(df: pd.DataFrame, all_models: list[str]) -> list[dict]:
    tert = _tertile(df)
    df = df.assign(_tert=tert)
    used: set[int] = set()

    def pick(score: pd.Series, ascending: bool, n_tert: int = 3) -> dict[int, int]:
        """Trả {tertile: row_index trong df} — điểm cao/thấp nhất CHƯA dùng, mỗi tertile 1 origin."""
        out = {}
        order = score.sort_values(ascending=ascending).index
        for t in range(n_tert):
            cand = [i for i in order if df.loc[i, "_tert"] == t and i not in used]
            if cand:
                out[t] = cand[0]
                used.add(cand[0])
        return out

    themes = []

    def add(name: str, reason: str, picks: dict[int, int]) -> None:
        if picks:
            themes.append({"name": name, "reason": reason, "picks": picks})

    med_dist = (df["rv60_bp"] - df.groupby(df["_tert"])["rv60_bp"].transform("median")).abs()
    add("Representative (typical)", "origin gần trung vị rv60 của tertile — ngày điển hình, không cực đoan", pick(med_dist, True))
    add("Good predictions", "avg |pred-actual| h3 trên 8 model nhỏ nhất trong tertile — model dự báo sát", pick(df["avg_abs_err_h3"], True))
    add("Bad predictions", "avg |pred-actual| h3 trên 8 model lớn nhất trong tertile — model đoán trật nhiều", pick(df["avg_abs_err_h3"], False))
    add("High model disagreement", "std giữa dự báo h3 của 8 model lớn nhất trong tertile", pick(df["disagreement_h3"], False))
    add("Low model disagreement", "std giữa dự báo h3 của 8 model nhỏ nhất trong tertile — model đồng thuận (đúng hay sai)",
        pick(df["disagreement_h3"], True))
    add("Strong upward move", "actual h3 (USD) dương lớn nhất trong tertile", pick(df["actual_h3"], False))
    add("Strong downward move", "actual h3 (USD) âm lớn nhất trong tertile", pick(df["actual_h3"], True))
    add("Flat / quiet period", "|actual h3| nhỏ nhất trong tertile — giá gần như đi ngang dù mức vol nào",
        pick(df["actual_h3"].abs(), True))
    rev = df[df["reversal_h3"]]
    if len(rev):
        add("Reversal pattern", "dấu actual h1 khác dấu actual h3 (đảo chiều trong 3 phút), |actual h3| lớn nhất trong tertile",
            pick(df["actual_h3"].abs().where(df["reversal_h3"]), False))
    mono = df[df["monotonic_h3"]]
    if len(mono):
        add("Monotonic trend", "actual h1→h2→h3 cùng dấu, độ lớn tăng dần — xu hướng mượt, |actual h3| lớn nhất trong tertile",
            pick(df["actual_h3"].abs().where(df["monotonic_h3"]), False))
    if "champion_gain_h3" in df:
        add("Champion (xgb) beats E0 clearly", "xgb sát actual hơn hẳn so với đoán 0 (E0) — gain h3 lớn nhất trong tertile",
            pick(df["champion_gain_h3"], False))
        add("Champion (xgb) underperforms", "xgb sai nhiều hơn cả đoán 0 (E0) — gain h3 nhỏ nhất (âm nhiều) trong tertile",
            pick(df["champion_gain_h3"], True))
    if "lstm_vs_tree_div_h3" in df:
        add("LSTM vs tree-model divergence", "|dự báo LSTM − trung bình dự báo tree-model| h3 lớn nhất trong tertile",
            pick(df["lstm_vs_tree_div_h3"].abs(), False))
    if "feat_vs_ref_div_h3" in df:
        add("Feature-rich vs B0/S0 reference divergence", "|trung bình dự báo 4 model đầy đủ feature − trung bình B0-306/B0*| h3 lớn nhất trong tertile",
            pick(df["feat_vs_ref_div_h3"].abs(), False))
    remaining = [i for i in df.index if i not in used]
    if remaining:
        rng = np.random.default_rng(8587)  # selection_seed của project — chỉ để chọn origin trình bày, không ảnh hưởng metric/model
        rem_df = df.loc[remaining]
        picks = {}
        for t in range(3):
            cand = rem_df.index[rem_df["_tert"] == t].tolist()
            if cand:
                picks[t] = cand[int(rng.integers(len(cand)))]
                used.add(picks[t])
        add("Broad coverage (stratified sample)", "origin còn lại, lấy mẫu ngẫu nhiên (seed 8587) mỗi tertile — bổ sung độ phủ, tránh chỉ chọn ca cực đoan",
            picks)
    return themes[:15]


# ----------------------------------------------------------------------------- vẽ 1 figure (2×3)
def plot_figure(df: pd.DataFrame, theme: dict, fig_id: int, top_models: list[str], bottom_models: list[str],
                champion: str, out_path: Path) -> list[dict]:
    picks = theme["picks"]
    cols = sorted(picks)  # 0=Low, 1=Medium, 2=High (thiếu tertile nào thì bấy nhiêu cột)
    n_col = len(cols)
    fig, axes = plt.subplots(2, n_col, figsize=(4.6 * n_col, 7.4), dpi=170, squeeze=False)
    rows_meta = []
    for col_i, t in enumerate(cols):
        row = df.loc[picks[t]]
        actual = [0.0] + [row[f"actual_h{h}"] for h in HORIZONS]
        ts_str = pd.Timestamp(int(row["ts"]), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
        title = f"{ts_str}\n{REGIME_LABEL[t]} vol — rv60 {row['rv60_bp']:.1f} bp"
        for ridx, (models, ax) in enumerate(((top_models, axes[0, col_i]), (bottom_models, axes[1, col_i]))):
            ax.axhline(0.0, color="#8a8a86", linestyle="--", linewidth=1.2, label="E0 (ŷ=0)", zorder=1)
            ax.plot(X, actual, color=ACTUAL_STYLE[0], linewidth=2.4, marker="o", markersize=5, label="Actual", zorder=5)
            for m in models:
                col = f"pred_{m}_h1"
                if col not in row:
                    continue
                y = [0.0] + [row[f"pred_{m}_h{h}"] for h in HORIZONS]
                c, mk = MODEL_STYLE[m]
                lbl = MODEL_LABEL[m] + (" [champion]" if m == champion else "")
                ax.plot(X, y, color=c, linewidth=1.6, marker=mk, markersize=5, alpha=0.9, label=lbl, zorder=4)
            ax.set_xticks(X); ax.set_xticklabels(["t", "t+1", "t+2", "t+3"])
            ax.grid(True, **GRID_KW); ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if col_i == 0:
                ax.set_ylabel("$P̂_{t+h} - C_t$ (USD)" if ridx == 0 else "$P̂_{t+h} - C_t$ (USD)")
            if ridx == 0:
                ax.set_title(title, fontsize=9.5)
        rows_meta.append({"col": col_i, "tertile": REGIME_LABEL[t], "row": row})
    axes[0, 0].annotate("Tree / ensemble", xy=(-0.32, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center",
                        fontsize=11, fontweight="bold")
    axes[1, 0].annotate("LSTM / reference", xy=(-0.32, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center",
                        fontsize=11, fontweight="bold")
    handles, labels = [], []
    for ax in (axes[0, 0], axes[1, 0]):
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Figure {fig_id:02d} — {theme['name']}  (FINAL TEST, champion = {MODEL_LABEL.get(champion, champion)})", fontsize=13, y=1.01)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.98))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return rows_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p0_ml_lstm.json")
    ap.add_argument("--out", default="reports/figures/final_test_paths")
    args = ap.parse_args()

    from p0.config import RunConfig

    cfg = RunConfig.load(args.config)
    exp = cfg.exp_dir
    sentinel = exp / "final" / "TEST_SENTINEL.json"
    if not sentinel.exists() or json.loads(sentinel.read_text(encoding="utf-8")).get("status") != "completed":
        sys.exit(f"FINAL TEST chưa hoàn tất ({sentinel}) — chạy `python run.py final --config {args.config}` trước.")
    index, preds = _load_final(exp)
    all_models = list(preds)  # CHỈ model thực sự có prediction FINAL TEST — không suy đoán/bịa thêm
    champion = index.get("champion") or "xgb"
    top_models = _series_available(preds, TOP_MODELS)
    bottom_models = _series_available(preds, BOTTOM_MODELS)
    print(f"models có prediction FINAL TEST: {all_models}")
    print(f"hàng trên (tree/ensemble): {top_models} | hàng dưới (khác/tham chiếu): {bottom_models} | champion={champion}")

    print("đọc data thật (Store, CPU, không train) để dựng lại actual/vol ...")
    store, _folds, final_fold, _rep = load_store(cfg)
    df = build_frame(store, index, preds)
    df = _add_metrics(df, all_models)

    themes = pick_themes(df, all_models)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for i, theme in enumerate(themes, start=1):
        out_path = out_dir / f"test_paths_{i:02d}.png"
        rows_meta = plot_figure(df, theme, i, top_models, bottom_models, champion, out_path)
        print(f"  {out_path.name}: {theme['name']} ({len(rows_meta)} origin)")
        for rm in rows_meta:
            r = rm["row"]
            index_rows.append({
                "figure_id": i, "figure_file": out_path.name, "theme": theme["name"], "reason": theme["reason"],
                "column": rm["col"], "volatility_regime": rm["tertile"], "timestamp_utc": pd.Timestamp(int(r["ts"]), unit="s", tz="UTC").isoformat(),
                "origin_idx": int(r["origin_idx"]), "rv60_bp": round(float(r["rv60_bp"]), 4),
                "actual_h1_usd": round(float(r["actual_h1"]), 4), "actual_h2_usd": round(float(r["actual_h2"]), 4),
                "actual_h3_usd": round(float(r["actual_h3"]), 4),
            })
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_csv(out_dir / "index.csv", index=False)
    (out_dir / "index.json").write_text(json.dumps(index_rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(themes)} figure, {len(index_rows)} origin -> {out_dir}")
    print(f"index -> {out_dir / 'index.csv'} / index.json")


if __name__ == "__main__":
    main()
