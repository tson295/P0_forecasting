"""§7.3 Figure.

**Fig P — forecast path** của MỘT origin (x = t, t+1, t+2, t+3; y = thay đổi giá so với C_t):
- Sau mỗi model: win vs champion trên 3 origin ở 3 ngày VAL vol thấp/trung bình/cao; + Fig HM (2 heatmap 15 ô).
- Final: heatmap TEST mọi model (khối 6h × h) + Fig P mọi model trên 3 origin TEST (vol thấp/trung vị/cao).
Actual luôn đen; E0 = đường ngang 0. Origin chọn theo quy tắc cố định (ngày/khối + giờ), KHÔNG theo error/prediction.

**Fig T — trajectory** (3 ảnh độc lập h = 1, 2, 3), chạy dọc TOÀN BỘ khoảng evaluation theo thời gian, vẽ **giá BTC thô**:
`actual_h(t) = C[t+h]`, `pred_h(t) = C[t]·exp(ŷ_h(t))`; trục x = timestamp **t+h**. Chỉ vẽ ở đúng hai chỗ đã có figure:
sau khi win_m so với champion (VAL) và ở Final (TEST). VAL không nối đường xuyên qua khoảng trống giữa các fold
(mỗi fold một đoạn riêng, vạch đứt xám ở ranh giới).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import HORIZONS  # noqa: E402
from .harness import Store  # noqa: E402
from .metrics import price_from_logret  # noqa: E402
from .palette import CHAMP_STYLE, GROUP_A, GROUP_B, INK, LABEL, MUTED, WIN_STYLE, style  # noqa: E402
from .split import Fold, Partition  # noqa: E402

STEPS = (0,) + HORIZONS  # trục x: t, t+1, t+2, t+3
XLAB = ["t"] + [f"t+{h}" for h in HORIZONS]


@dataclass
class OriginPick:
    """Một origin đại diện cho một chế độ biến động; idx = chỉ số origin trên lưới."""

    label: str
    idx: int


def _stamp(ts: int) -> str:
    return pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%m-%d %H:%M")


def day_vol(store: Store, fold: Fold) -> float:
    idx = fold.val.origins(store.ts, store.eligible)
    r1 = np.diff(np.log(store.close[idx.min():idx.max() + 1]))
    return float(np.nanstd(r1))


def select_vol_origins(store: Store, folds: list[Fold], start_hour: int = 12) -> list[OriginPick]:
    """3 ngày VAL khác nhau theo std r1 trong ngày: min / trung vị / max; mỗi ngày lấy origin đầu tiên ≥ start_hour:00 UTC.
    Quy tắc cố định, không phụ thuộc prediction/error."""
    vols = sorted(((day_vol(store, f), i, f) for i, f in enumerate(folds)), key=lambda x: (x[0], x[1]))
    picks = [vols[0], vols[len(vols) // 2], vols[-1]]
    out = []
    for (v, _, f), tag in zip(picks, ("vol thấp", "vol trung bình", "vol cao")):
        idx = f.val.origins(store.ts, store.eligible)
        k = min(int(np.searchsorted(store.ts[idx], f.val.start + start_hour * 3600)), len(idx) - 1)
        out.append(OriginPick(f"{_stamp(store.ts[idx[k]])} ({tag}, std r1 ngày {v * 1e4:.1f}bp)", int(idx[k])))
    return out


def select_vol_origins_test(store: Store, test: Partition, n: int = 60) -> list[OriginPick]:
    """TEST: chia thành khối n origin không chồng nhau, std r1 mỗi khối → thấp nhất / trung vị / cao nhất;
    origin đại diện = origin ĐẦU của khối (cố định, không theo error)."""
    idx = test.origins(store.ts, store.eligible)
    blocks = [idx[i:i + n] for i in range(0, len(idx) - n + 1, n)]
    vols = [float(np.nanstd(np.diff(np.log(store.close[b[0]:b[-1] + 1])))) for b in blocks]
    order = np.argsort(vols, kind="stable")
    picks = [order[0], order[len(order) // 2], order[-1]]
    return [OriginPick(f"{_stamp(store.ts[blocks[p][0]])} ({tag}, std r1 khối {vols[p] * 1e4:.1f}bp)", int(blocks[p][0]))
            for p, tag in zip(picks, ("vol thấp", "vol trung bình", "vol cao"))]


def _preds_on(store: Store, idx: np.ndarray, preds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Lấy ŷ (n,3) cho các origin idx từ danh sách (idx_val, yhat) của các fold."""
    for idx_val, yhat in preds:
        if len(idx_val) == 0:
            continue
        pos = np.searchsorted(idx_val, idx)
        if not np.all(pos < len(idx_val)):
            continue
        if np.array_equal(idx_val[pos], idx):
            return yhat[pos]
    return None


def _path(c_t: float, yhat_row: np.ndarray | None, c_future_row: np.ndarray | None = None) -> np.ndarray:
    """Đường dự báo/thực tế dạng thay đổi giá so với C_t: [0, ·−C_t ×3]. yhat → P̂ = C_t·exp(ŷ)."""
    if c_future_row is not None:
        return np.r_[0.0, np.asarray(c_future_row, float) - c_t]
    p_hat = price_from_logret(np.array([c_t]), np.asarray(yhat_row, float)[None, :])[0]
    return np.r_[0.0, p_hat - c_t]


def _panel(ax, store: Store, pick: OriginPick, series: list[tuple[str, list, str, str]], show_legend: bool) -> None:
    idx = np.array([pick.idx], dtype=np.int64)
    c_t, c_future, _ = store.targets(idx)
    x = np.arange(len(STEPS))
    ax.axhline(0.0, color=MUTED, ls="--", lw=0.9, label="E0 (P̂ = C_t) = 0")
    ax.plot(x, _path(float(c_t[0]), None, c_future[0]), color=INK, lw=1.4, marker="o", ms=5, label="actual (C_(t+h) − C_t)")
    for label, preds, color, marker in series:
        yhat = _preds_on(store, idx, preds)
        if yhat is None:
            continue
        ax.plot(x, _path(float(c_t[0]), yhat[0]), color=color, lw=1.2, ls="-", marker=marker, ms=6.5, alpha=0.9, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB)
    ax.set_xlabel("bước dự báo từ origin t")
    ax.set_title(f"{pick.label}  |  C_t = {float(c_t[0]):,.0f} USD", fontsize=8.5)
    if show_legend:
        ax.set_ylabel("thay đổi giá so với C_t (USD)")
        ax.legend(fontsize=7, loc="best")


def fig_path(store: Store, picks: list[OriginPick], series: list[tuple[str, list, str, str]], out: Path, title: str) -> None:
    """Một ảnh = 3 panel, mỗi panel một origin: actual vs prediction của win/champion theo t → t+3."""
    fig, axes = plt.subplots(1, len(picks), figsize=(5.6 * len(picks), 4.4))
    axes = np.atleast_1d(axes)
    for k, pick in enumerate(picks):
        _panel(axes[k], store, pick, series, show_legend=(k == 0))
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def final_fig_paths(store: Store, picks: list[OriginPick], preds_by_model: dict[str, list], out: Path) -> None:
    """Final: 2 hàng (nhóm A tree + ensemble; nhóm B TimesFM/AutoTS/LSTM + reference) × 3 origin; mỗi model màu/marker cố định."""
    groups = [("nhóm A: tree + ensemble", [m for m in GROUP_A if m in preds_by_model]),
              ("nhóm B: TimesFM / AutoTS / LSTM + reference", [m for m in GROUP_B if m in preds_by_model])]
    fig, axes = plt.subplots(2, len(picks), figsize=(5.6 * len(picks), 8.6))
    axes = np.atleast_2d(axes)
    for gi, (gname, group) in enumerate(groups):
        series = [(LABEL.get(m, m), preds_by_model[m], *style(m)[:2]) for m in group]
        for k, pick in enumerate(picks):
            _panel(axes[gi, k], store, pick, series, show_legend=(k == 0))
            axes[gi, k].set_title(f"{gname}\n{pick.label}", fontsize=8)
    fig.suptitle("Final — forecast path mọi model trên TEST: x = t → t+3, y = thay đổi giá so với C_t; actual đen; ≤ 8 màu mỗi panel",
                 fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig T: trajectory theo thời gian
def _dt(ts) -> np.ndarray:
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).tz_localize(None).to_numpy()


def _traj_panel(ax, store: Store, h: int, series: list[tuple[str, list, str, str]], mark_folds: bool) -> None:
    """Một panel: actual C_(t+h) (đen) + prediction của các series, x = thời điểm t+h. Mỗi fold một đoạn, không nối qua nhau."""
    segs = series[0][1] if series else []
    for k, (idx, _) in enumerate(segs):
        x = _dt(store.ts[idx] + 60 * h)
        ax.plot(x, store.close[idx + h], color=INK, lw=0.9, label=f"actual C_(t+{h})" if k == 0 else None)
        if mark_folds and k:
            ax.axvline(x[0], color=MUTED, lw=0.7, ls=":", alpha=0.8)
    for si, (label, preds, color, marker) in enumerate(series):
        for k, (idx, yhat) in enumerate(preds):
            x = _dt(store.ts[idx] + 60 * h)
            p_hat = price_from_logret(store.close[idx], yhat)[:, h - 1]
            step = max(1, len(x) // 30)
            phase = (si * step) // max(1, len(series))  # lệch pha marker theo series để series vẽ sau không che series trước
            ax.plot(x, p_hat, color=color, lw=0.9, alpha=0.85, marker=marker, ms=3.5, markevery=(phase, step),
                    label=label if k == 0 else None)
    ax.set_ylabel("giá BTC (USD)")
    ax.margins(x=0.01)


def fig_trajectory(store: Store, h: int, series: list[tuple[str, list, str, str]], out: Path, title: str,
                   mark_folds: bool = True) -> None:
    """Một ảnh cho một horizon: toàn bộ khoảng evaluation, actual vs prediction theo thời điểm t+h."""
    fig, ax = plt.subplots(figsize=(16.5, 5.0))
    _traj_panel(ax, store, h, series, mark_folds)
    ax.set_xlabel("thời điểm được dự báo t+h (UTC)")
    ax.legend(fontsize=8, loc="best", ncol=3)
    fig.suptitle(title, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def final_fig_trajectory(store: Store, h: int, preds_by_model: dict[str, list], out: Path) -> None:
    """Final: cùng định nghĩa Fig T trên TEST, tách 2 nhóm như Fig P để mỗi panel ≤ 8 màu."""
    groups = [("nhóm A: tree + ensemble", [m for m in GROUP_A if m in preds_by_model]),
              ("nhóm B: TimesFM / AutoTS / LSTM + reference", [m for m in GROUP_B if m in preds_by_model])]
    fig, axes = plt.subplots(2, 1, figsize=(16.5, 9.5), sharex=True)
    for gi, (gname, group) in enumerate(groups):
        series = [(LABEL.get(m, m), preds_by_model[m], *style(m)[:2]) for m in group]
        if not series:
            axes[gi].axis("off")
            continue
        _traj_panel(axes[gi], store, h, series, mark_folds=False)
        axes[gi].set_title(gname, fontsize=8.5)
        axes[gi].legend(fontsize=7, loc="best", ncol=4)
    axes[1].set_xlabel("thời điểm được dự báo t+h (UTC)")
    fig.suptitle(f"Fig T{h} — trajectory TEST của mọi model: actual C_(t+{h}) (đen) vs P̂_(t+{h}) = C_t·exp(ŷ_{h}); giá BTC thô",
                 fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
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
