"""Fig P mọi model trên VAL — Y HỆT `final_fig_paths` (§7.3): mỗi ảnh 2 hàng (nhóm A / nhóm B) × 3 origin.

Khác `cmd_final` đúng hai chỗ, vì prediction TEST không còn (`experiments/runs/` bị gitignore):
  - dữ liệu = prediction VAL đã lưu `wins/<model>_seed0.npz` (selection_seed) thay vì pred_test.npz;
  - origin chọn trên VAL bằng CHÍNH quy tắc `select_vol_origins_test` (khối 60 origin, std r1, lấy origin
    ĐẦU khối), mở rộng từ 1 bộ 3 lên N bộ 3: xếp mọi khối theo vol, lấy 3N khối rải đều, chia 3 tầng
    thấp/trung bình/cao → ảnh k = (thấp[k], trung bình[k], cao[k]). Khối không bắc qua ranh giới fold.
Layout/màu/marker/nhóm A-B lấy nguyên từ `p0.plots` + `p0.palette` — không vẽ lại gì mới.

    python scripts/fig_val_paths_all_models.py [--n 10] [--config configs/p0_15d.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from p0.cli import load_store  # noqa: E402
from p0.config import RunConfig  # noqa: E402
from p0.logs import load_preds  # noqa: E402
from p0.palette import GROUP_A, GROUP_B, LABEL, style  # noqa: E402
from p0.plots import OriginPick, _panel, _stamp  # noqa: E402

TAGS = ("vol thấp", "vol trung bình", "vol cao")


def val_pick_triples(store, folds, n_img: int, block: int = 60) -> list[list[OriginPick]]:
    """`n_img` bộ 3 origin. Mọi khối VAL xếp theo std r1 → lấy 3·n_img khối rải đều → chia 3 tầng
    thấp / trung bình / cao (mỗi tầng n_img khối) → ảnh k ghép (thấp[k], trung bình[k], cao[k])."""
    blocks = []
    for f in folds:
        idx = f.val.origins(store.ts, store.eligible)
        blocks += [idx[i:i + block] for i in range(0, len(idx) - block + 1, block)]
    vols = [float(np.nanstd(np.diff(np.log(store.close[b[0]:b[-1] + 1])))) for b in blocks]
    order = np.argsort(vols, kind="stable")
    k = 3 * n_img
    if k > len(order):
        raise SystemExit(f"chỉ có {len(order)} khối {block} origin trên VAL, không đủ cho {k} origin phân biệt "
                         f"— giảm --n hoặc --block")
    take = np.linspace(0, len(order) - 1, k).round().astype(int)
    for i in range(1, k):  # linspace có thể trùng khi k gần len(order) → đẩy tới cho phân biệt, vẫn giữ thứ tự vol
        take[i] = max(int(take[i]), int(take[i - 1]) + 1)
    assert take[-1] <= len(order) - 1 and len(np.unique(take)) == k
    picked = [int(order[o]) for o in take]  # đã sắp theo vol tăng dần
    tiers = [picked[i * n_img:(i + 1) * n_img] for i in range(3)]  # thấp / trung bình / cao
    return [[OriginPick(f"{_stamp(store.ts[blocks[p][0]])} ({tag}, std r1 khối {vols[p] * 1e4:.1f}bp)", int(blocks[p][0]))
             for tag, p in zip(TAGS, (tiers[0][k], tiers[1][k], tiers[2][k]))] for k in range(n_img)]


def fig_paths(store, picks: list[OriginPick], preds_by_model: dict, out: Path, k: int, n: int) -> None:
    """Bản sao `plots.final_fig_paths`, chỉ đổi suptitle TEST → VAL và thêm số thứ tự ảnh."""
    groups = [("nhóm A: tree + ensemble", [m for m in GROUP_A if m in preds_by_model]),
              ("nhóm B: TimesFM / AutoTS / LSTM + reference", [m for m in GROUP_B if m in preds_by_model])]
    fig, axes = plt.subplots(2, len(picks), figsize=(5.6 * len(picks), 8.6))
    axes = np.atleast_2d(axes)
    for gi, (gname, group) in enumerate(groups):
        series = [(LABEL.get(m, m), preds_by_model[m], *style(m)[:2]) for m in group]
        for c, pick in enumerate(picks):
            _panel(axes[gi, c], store, pick, series, show_legend=(c == 0))
            axes[gi, c].set_title(f"{gname}\n{pick.label}", fontsize=8)
    fig.suptitle(f"VAL [{k}/{n}] — forecast path mọi model: x = t → t+3, y = thay đổi giá so với C_t; "
                 "actual đen; ≤ 8 màu mỗi panel", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p0_15d.json"), help="config của run có wins/ (mặc định: vòng 15 ngày)")
    ap.add_argument("--n", type=int, default=10, help="số ẢNH (mỗi ảnh 3 origin)")
    ap.add_argument("--block", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = RunConfig.load(args.config)
    exp = cfg.exp_dir
    out_dir = Path(args.out) if args.out else exp / "summary" / "val_paths"
    store, folds, _, _ = load_store(cfg)
    preds_by_model, missing = {}, []
    for m in GROUP_A + GROUP_B:
        p = exp / "wins" / f"{m}_seed0.npz"
        (preds_by_model.__setitem__(m, load_preds(p)) if p.exists() else missing.append(m))
    if missing:
        print(f"THIẾU prediction VAL (không vẽ được): {', '.join(missing)}")
    print(f"model vẽ: {', '.join(preds_by_model)}")
    triples = val_pick_triples(store, folds, args.n, args.block)
    for k, picks in enumerate(triples, start=1):
        f = out_dir / f"fig_val_paths_all_models_{k:02d}.png"
        fig_paths(store, picks, preds_by_model, f, k, len(triples))
        print(f"[{k:2d}/{len(triples)}] " + "  |  ".join(p.label for p in picks))
    print(f"\n{len(triples)} ảnh × 3 origin = {3 * len(triples)} origin → {out_dir}")


if __name__ == "__main__":
    main()
