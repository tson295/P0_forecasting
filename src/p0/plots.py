"""§7.3 Figure: Fig H_h (win vs champion, 3 cửa sổ ở 3 ngày VAL vol thấp/trung bình/cao), Fig HM (2 heatmap 15 ô),
Final: heatmap TEST mọi model (khối 6h × h) + Fig H_h mọi model. Actual luôn đen; không vẽ chuỗi dự báo liên tục."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import HORIZONS  # noqa: E402
from .harness import Store  # noqa: E402
from .metrics import price_from_logret  # noqa: E402
from .palette import CHAMP_STYLE, GROUP_A, GROUP_B, INK, LABEL, MUTED, WIN_STYLE, style  # noqa: E402
from .split import Fold, Partition  # noqa: E402


@dataclass
class Window:
    label: str
    idx: np.ndarray  # chỉ số origin trên lưới


def day_vol(store: Store, fold: Fold) -> float:
    idx = fold.val.origins(store.ts, store.eligible)
    r1 = np.diff(np.log(store.close[idx.min():idx.max() + 1]))
    return float(np.nanstd(r1))


def select_vol_windows(store: Store, folds: list[Fold], start_hour: int = 12, n: int = 60) -> list[Window]:
    """3 ngày VAL khác nhau theo std r1: min / trung vị / max; cửa sổ n origin từ start_hour UTC."""
    vols = [(day_vol(store, f), f) for f in folds]
    order = sorted(vols, key=lambda x: x[0])
    picks = [order[0], order[len(order) // 2], order[-1]]
    tags = ["vol thấp", "vol trung bình", "vol cao"]
    out = []
    for (v, f), tag in zip(picks, tags):
        start = f.val.start + start_hour * 3600
        idx = Partition(start, start + n * 60 + 180).origins(store.ts, store.eligible)[:n]
        out.append(Window(f"{f.name.split('_')[-1]} {start_hour:02d}:00 ({tag}, std r1 {v * 1e4:.1f}bp)", idx))
    return out


def select_vol_windows_test(store: Store, test: Partition, n: int = 60) -> list[Window]:
    """TEST: 3 cửa sổ n origin không chồng nhau theo std r1 của cửa sổ: thấp nhất / trung vị / cao nhất."""
    idx = test.origins(store.ts, store.eligible)
    blocks = [idx[i:i + n] for i in range(0, len(idx) - n + 1, n)]
    vols = [float(np.nanstd(np.diff(np.log(store.close[b[0]:b[-1] + 1])))) for b in blocks]
    order = np.argsort(vols)
    picks = [order[0], order[len(order) // 2], order[-1]]
    tags = ["vol thấp", "vol trung bình", "vol cao"]
    out = []
    for p, tag in zip(picks, tags):
        b = blocks[p]
        import pandas as pd

        t0 = pd.Timestamp(int(store.ts[b[0]]), unit="s", tz="UTC").strftime("%m-%d %H:%M")
        out.append(Window(f"{t0} ({tag}, std r1 {vols[p] * 1e4:.1f}bp)", b))
    return out


def _preds_on(store: Store, idx: np.ndarray, preds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Lấy ŷ (n,3) cho các origin idx từ danh sách (idx_val, yhat) của các fold."""
    for idx_val, yhat in preds:
        pos = np.searchsorted(idx_val, idx)
        if len(idx_val) == 0 or not np.all(pos < len(idx_val)):
            continue
        if np.array_equal(idx_val[pos], idx):
            return yhat[pos]
    return None


def fig_h(store: Store, h: int, windows: list[Window], series: list[tuple[str, list, str, str]], out: Path, title: str) -> None:
    """series: (label, preds list[(idx_val, yhat)], color, marker). Một ảnh = 3 panel (3 cửa sổ)."""
    fig, axes = plt.subplots(1, len(windows), figsize=(6.2 * len(windows), 4.4))
    axes = np.atleast_1d(axes)
    for k, w in enumerate(windows):
        ax = axes[k]
        c_t, c_future, _ = store.targets(w.idx)
        x = np.arange(len(w.idx))
        ax.plot(x, c_future[:, h - 1], color=INK, lw=1.0, marker="o", ms=3.5, label=f"giá thật C_(t+{h})")
        ax.plot(x, c_t, color=MUTED, ls="--", lw=0.9, label="E0 (P̂ = C_t)")
        for label, preds, color, marker in series:
            yhat = _preds_on(store, w.idx, preds)
            if yhat is None:
                continue
            p_hat = price_from_logret(c_t, yhat)[:, h - 1]
            ax.plot(x, p_hat, color=color, ls="none", marker=marker, ms=6, alpha=0.9, label=label)
        ax.set_title(f"{w.label} — h={h}", fontsize=9)
        ax.set_xlabel("origin t (phút trong cửa sổ)")
        if k == 0:
            ax.set_ylabel("USD")
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _heatmap(ax, mat, row_labels, title, vmax=0.3):
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"h={h}" for h in HORIZONS], fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=8)
    return im


def fig_hm(win_tab: np.ndarray, champ_tab: np.ndarray, row_labels: list[str], win_label: str, champ_label: str, footer: str, out: Path,
           suptitle: str = "Fig HM — Gain vs E0 (pp) từ RMSE̅ mean 3 seed, cùng thang màu") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    _heatmap(axes[0], win_tab, row_labels, f"win = {win_label}")
    im = _heatmap(axes[1], champ_tab, row_labels, f"champion = {champ_label}")
    fig.subplots_adjust(left=0.09, right=0.86, top=0.82, bottom=0.16, wspace=0.45)
    cax = fig.add_axes([0.89, 0.16, 0.02, 0.64])
    fig.colorbar(im, cax=cax, label="Gain vs E0 (pp)")
    fig.suptitle(suptitle, fontsize=9)
    fig.text(0.5, 0.04, footer, ha="center", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def final_heatmaps(tables: dict[str, np.ndarray], block_labels: list[str], out: Path) -> None:
    keys = list(tables)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 4.2 * nrow + 1.2))
    axes = np.atleast_2d(axes)
    im = None
    for i, (ax, k) in enumerate(zip(axes.ravel(), keys)):
        im = _heatmap(ax, tables[k], block_labels if i % ncol == 0 else [""] * len(block_labels), f"{LABEL.get(k, k)}\nGain vs E0 (pp), TEST")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.07, right=0.90, top=1 - 1.1 / (4.2 * nrow + 1.2), bottom=0.06, wspace=0.25, hspace=0.45)
    if im is not None:
        cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
        fig.colorbar(im, cax=cax, label="Gain vs E0 (pp)")
    fig.suptitle("Final — heatmap TEST của mọi model: ô = khối 6 giờ × horizon; cùng thang màu", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def final_fig_h(store: Store, h: int, windows: list[Window], preds_by_model: dict[str, list], out: Path) -> None:
    """Hai hàng (nhóm A tree + ensemble; nhóm B TimesFM/AutoTS/LSTM + reference) × 3 cửa sổ; mỗi model màu/marker cố định."""
    groups = [("nhóm A: tree + ensemble", [m for m in GROUP_A if m in preds_by_model]),
              ("nhóm B: TimesFM / AutoTS / LSTM + reference", [m for m in GROUP_B if m in preds_by_model])]
    fig, axes = plt.subplots(2, len(windows), figsize=(6.2 * len(windows), 8.6))
    axes = np.atleast_2d(axes)
    for gi, (gname, group) in enumerate(groups):
        for k, w in enumerate(windows):
            ax = axes[gi, k]
            c_t, c_future, _ = store.targets(w.idx)
            x = np.arange(len(w.idx))
            ax.plot(x, c_future[:, h - 1], color=INK, lw=1.0, marker="o", ms=3.5, label=f"giá thật C_(t+{h})")
            ax.plot(x, c_t, color=MUTED, ls="--", lw=0.9, label="E0 (P̂ = C_t)")
            for m in group:
                yhat = _preds_on(store, w.idx, preds_by_model[m])
                if yhat is None:
                    continue
                col, mk, _ = style(m)
                ax.plot(x, price_from_logret(c_t, yhat)[:, h - 1], color=col, ls="none", marker=mk, ms=5.5, alpha=0.9, label=LABEL.get(m, m))
            ax.set_title(f"{gname} — {w.label} — h={h}", fontsize=8)
            ax.set_xlabel("origin t (phút trong cửa sổ)")
            if k == 0:
                ax.set_ylabel("USD")
                ax.legend(fontsize=6.5, loc="best")
    fig.suptitle(f"Final — Fig H{h} của mọi model trên TEST: P̂_(t+{h}) theo origin t; actual đen; ≤ 8 màu mỗi panel", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
