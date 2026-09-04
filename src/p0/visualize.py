"""§7.3 hậu kỳ (quyết định user 2026-09-03: KHÔNG vẽ trong đường chạy training/search).

`python run.py visualize --config <cfg>` dựng lại MỌI figure từ artifact đã lưu, không train, không inference:
- mỗi dòng so champion trong `champion_log.csv` (model m vs champion trước) → Fig P / Fig T_h / Fig HM từ
  `wins/<m>_seed0.npz` + `wins/<champion>_seed0.npz` + bảng RMSE̅/E0 trong `wins/*.json`;
- cặp cấu hình: TimesFM-LoRA native vs TimesFM-LoRA + XReg (`wins/tfm_lora_native.json` / `wins/tfm_lora_xreg.json`), AutoTS WR vs MR probe;
- Final (TEST): heatmap khối 6h × h, Fig P và Fig T_h mọi model từ `final/index.json` + `final/<key>.npz`.
Chỉ cần data (để dựng lại actual/giá) + artifact. Định nghĩa figure giữ nguyên `plots.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import HORIZONS
from .harness import Store
from .logs import load_preds
from .metrics import cell_metrics, e0_rmse, gain_pp
from .palette import LABEL
from .split import Fold


def _tab_e0(w: dict) -> np.ndarray:
    return gain_pp(np.asarray(w["rmse_mean"]), np.asarray(w["e0"]))


def _win(exp: Path, name: str) -> dict | None:
    p = exp / "wins" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _preds(exp: Path, name: str):
    p = exp / "wins" / f"{name}_seed0.npz"
    return load_preds(p) if p.exists() else None


def pair_figs(store: Store, folds: list[Fold], out: Path, left: tuple, right: tuple, tag: str, title: str, footer: str,
              prefixes: tuple[str, str] = ("win", "champion")) -> list[Path]:
    """Bộ figure cho MỘT cặp (Fig P + Fig T_h + Fig HM) → out/*_{tag}.png. left/right = (name, preds, tab_e0)."""
    (l_name, l_preds, l_tab), (r_name, r_preds, r_tab) = left, right
    if l_preds is None or r_preds is None:
        return []
    l_lab, r_lab = LABEL.get(l_name, l_name), LABEL.get(r_name, r_name)
    series = [(f"{prefixes[0]} = {l_lab}", l_preds, plots.WIN_STYLE[0], plots.WIN_STYLE[1]),
              (f"{prefixes[1]} = {r_lab}", r_preds, plots.CHAMP_STYLE[0], plots.CHAMP_STYLE[1])]
    picks = plots.select_vol_origins(store, folds)
    made = []
    f = out / f"fig_path_{tag}.png"
    plots.fig_path(store, picks, series, f, f"Fig P — {title}: x = t → t+3, y = thay đổi giá so với C_t")
    made.append(f)
    for h in HORIZONS:
        f = out / f"fig_traj_h{h}_{tag}.png"
        plots.fig_trajectory(store, h, series, f, f"Fig T{h} — trajectory VAL ({title}): actual C_(t+{h}) vs P̂_(t+{h}) = C_t·exp(ŷ_{h})")
        made.append(f)
    f = out / f"fig_HM_{tag}.png"
    plots.fig_hm(l_tab, r_tab, [fo.name.split("_")[-1] for fo in folds], l_lab, r_lab, footer, f, prefixes=prefixes)
    made.append(f)
    return made


def champion_figs(store: Store, folds: list[Fold], exp: Path, out: Path) -> list[Path]:
    """Một bộ figure cho mỗi lần so champion đã ghi trong champion_log.csv (bỏ dòng probe / ensemble)."""
    p = exp / "champion_log.csv"
    if not p.exists():
        return []
    log = pd.read_csv(p)
    made = []
    for _, r in log.iterrows():
        m, before, decision = str(r["model"]), str(r.get("champion_before", "") or ""), str(r.get("decision", ""))
        if decision.startswith("probe") or m.startswith("ensemble") or not before or before == "nan":
            continue
        wm, wc = _win(exp, m), _win(exp, before)
        if wm is None or wc is None:
            continue
        footer = f"win vs champion: {decision}" + (f" — MedianGain {float(r['MedianGain_vs_champion']):+.4f} (ε {float(r['eps_champion']):.4f})"
                                                  if pd.notna(r.get("MedianGain_vs_champion", np.nan)) else "")
        made += pair_figs(store, folds, out, (m, _preds(exp, m), _tab_e0(wm)), (before, _preds(exp, before), _tab_e0(wc)),
                          f"{m}_vs_champion", f"win vs champion ({m} vs {before})", footer)
    return made


def branch_figs(store: Store, folds: list[Fold], exp: Path, out: Path) -> list[Path]:
    made = []
    a, b = _win(exp, "tfm_lora_native"), _win(exp, "tfm_lora_xreg")
    if a and b:
        made += pair_figs(store, folds, out, ("tfm_lora_xreg", _preds(exp, "tfm_lora_xreg"), _tab_e0(b)), ("tfm_lora_native", _preds(exp, "tfm_lora_native"), _tab_e0(a)),
                          "tfm_lora_xreg_vs_native", "TimesFM-LoRA + XReg(F_best) vs TimesFM-LoRA native",
                          "cùng adapter LoRA đã freeze — TFM-final chọn bằng MedianGain > +ε_TFM", prefixes=("+XReg", "native"))
    a, b = _win(exp, "autots_wr"), _win(exp, "autots_mr")
    if a and b:
        made += pair_figs(store, folds, out, ("autots_wr", _preds(exp, "autots_wr"), _tab_e0(a)), ("autots_mr", _preds(exp, "autots_mr"), _tab_e0(b)),
                          "autots_wr_vs_autots_mr", "nhánh AutoTS: autots_wr (WindowRegression) vs autots_mr (MultivariateRegression)",
                          "branch vs branch — AutoTS-final chọn sau bake-off bằng metric project", prefixes=("nhánh", "nhánh"))
    return made


def final_figs(store: Store, final: Fold, exp: Path, out: Path) -> list[Path]:
    """Final: heatmap khối 6h × h, Fig P mọi model, Fig T_h mọi model — từ final/index.json + final/<key>.npz."""
    idx_path = exp / "final" / "index.json"
    if not idx_path.exists():
        return []
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    idx_test = final.val.origins(store.ts, store.eligible)
    c_t, c_future, _ = store.targets(idx_test)
    block = (store.ts[idx_test] - final.val.start) // (6 * 3600)
    blocks = sorted(set(block.tolist()))
    block_labels = [f"{pd.Timestamp(final.val.start + b * 21600, unit='s', tz='UTC').strftime('%m-%d %H')}h" for b in blocks]
    e0_blocks = np.array([e0_rmse(c_t[block == b], c_future[block == b]) for b in blocks])
    tables, preds_by_model = {}, {}
    for key in index["keys"]:
        p = exp / "final" / f"{key}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        idx, yhat = z["idx_0"], z["yhat_0"]
        if not np.array_equal(idx, idx_test):
            raise ValueError(f"final/{key}.npz: origin TEST không khớp data hiện tại")
        tables[key] = np.array([gain_pp(cell_metrics(c_t[block == b], c_future[block == b], yhat[block == b])["rmse"], e0_blocks[i])
                                for i, b in enumerate(blocks)])
        preds_by_model[key] = [(idx, yhat)]
    made = []
    if tables:
        f = out / "fig_final_heatmaps.png"
        plots.final_heatmaps(tables, block_labels, f)
        made.append(f)
        picks = plots.select_vol_origins_test(store, final.val)
        f = out / "fig_final_paths_all_models.png"
        plots.final_fig_paths(store, picks, preds_by_model, f)
        made.append(f)
        for h in HORIZONS:
            f = out / f"fig_final_traj_h{h}_all_models.png"
            plots.final_fig_trajectory(store, h, preds_by_model, f)
            made.append(f)
    return made


def regenerate_all(store: Store, folds: list[Fold], final: Fold, exp: Path, out: Path | None = None) -> list[Path]:
    out = out or (exp / "summary")
    out.mkdir(parents=True, exist_ok=True)
    return champion_figs(store, folds, exp, out) + branch_figs(store, folds, exp, out) + final_figs(store, final, exp, out)
