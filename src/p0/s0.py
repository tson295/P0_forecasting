"""S0_m KHOÁ + overlap audit + Candidate_m — vòng expanded-data (quyết định user 2026-09-03, hiệu chỉnh 2026-09-04).

- S0_m = B0* ∪ F_old_m, dựng từ ARTIFACT thắng vòng 15 ngày (`<prev_run_dir>/wins/<m>.json` + `b0_star.json`), không gõ tay.
  **TOÀN BỘ S0_m là khoá**: `locked_b0 == S0.b0` (mọi cột B0*), `locked_ext == F_old_m` (mọi cột ext thắng cũ). Không phép toán
  nào của pipeline bỏ được cột khoá (B0 không bao giờ bị xoá; ext khoá không bị prune PI, không `without_ext`). Prune PI chỉ xét
  cột ext MỚI. AutoTS: mỗi nhánh probe (autots_wr / autots_mr) kế thừa ĐÚNG bộ thắng của nhánh đó. TimesFM: S0 = ∅.
  Phân biệt tên: S0_m = tập xuất phát khoá; F_raw_m / F_pruned_m / F_best_m = tập do vòng tìm kiếm mới sinh ra.
- Candidate_m = C_short \\ overlap(C_short, S0_m), tính RIÊNG cho từng model và lưu riêng (`s0/candidates_<m>.json`).
  overlap = cột C_short đã có trong S0_m của CHÍNH model đó: trùng tên, hoặc khác tên nhưng giá trị giống hệt tại cùng timestamp
  (kiểm bằng số: `max|a−b| ≤ ident_rtol·std`). KHÔNG bỏ vì: tương quan cao, xấp xỉ, cửa sổ gần nhau, cùng họ, có trong B0-306
  nhưng KHÔNG trong B0* của model, hay từng là candidate cũ nhưng không trong S0. Cùng indicator khác lag KHÔNG phải trùng
  (khác timestamp). Tương quan cao / trùng giá trị giữa các cột C_short với nhau chỉ là chẩn đoán (báo cáo, không bỏ).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features_ext import Candidate
from .features_short import SHORT_BY_NAME, SHORT_COLUMNS, short_pool_report
from .harness import ColSet, Store

# model → file thắng của vòng trước; TimesFM không kế thừa
PREV_WINNER = {"lgbm": "lgbm.json", "xgb": "xgb.json", "cat": "cat.json", "xgbrf": "xgbrf.json", "lstm": "lstm.json",
               "autots_wr": "autots_wr.json", "autots_mr": "autots_mr.json"}
S0_MODELS = tuple(PREV_WINNER) + ("tfm",)


def prev_winner(prev_dir: Path, model: str) -> dict:
    p = Path(prev_dir) / "wins" / PREV_WINNER[model]
    if not p.exists():
        raise FileNotFoundError(f"thiếu artifact thắng của vòng trước cho {model}: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def prev_dropped(prev_dir: Path, model: str) -> list[str]:
    """Cột DROP của vòng trước (keepdrop_<m>.csv) — chỉ để kiểm tra C_short không chứa định nghĩa cũ."""
    p = Path(prev_dir) / f"keepdrop_{model}.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    return [c for cols in d.loc[d["decision"] == "DROP", "columns"] for c in str(cols).split("|")]


def s0_for(model: str, prev_dir: Path | None, b0_star: ColSet | None = None) -> ColSet:
    """S0_m: từ winner vòng trước (locked_b0 = B0*, locked_ext = F_old_m); TimesFM = ∅; không có vòng trước (smoke) → B0*."""
    if model == "tfm":
        return ColSet((), ())
    if prev_dir is None:
        if b0_star is None:
            raise ValueError("không có prev_run_dir lẫn B0* — không dựng được S0")
        return ColSet(tuple(b0_star.b0))
    w = prev_winner(prev_dir, model)
    cs = ColSet.from_dict(w["colset"])
    star_path = Path(prev_dir) / "b0_star.json"
    if not star_path.exists():
        raise FileNotFoundError(f"thiếu {star_path} — B0* của vòng trước là bắt buộc để kiểm tra S0_{model}")
    star = ColSet.load(star_path)
    if tuple(cs.b0) != tuple(star.b0):
        raise ValueError(f"{model}: cột B0 trong wins/{PREV_WINNER[model]} khác b0_star.json của vòng trước")
    return ColSet(cs.b0, cs.ext, cs.ext)


def assert_s0_schema(cs: ColSet, d: dict, model: str) -> None:
    """Artifact S0 phải khoá TOÀN BỘ: locked_b0 == b0 và locked_ext == ext (hard invariant — artifact lệch = malformed)."""
    if list(d.get("locked_b0", [])) != list(cs.b0) or list(d.get("locked_ext", [])) != list(cs.ext):
        raise ValueError(f"S0_{model} malformed: locked_b0 phải == b0 và locked_ext phải == ext (toàn bộ S0 là khoá)")
    if tuple(cs.locked_ext) != tuple(cs.ext):
        raise ValueError(f"S0_{model} malformed: ext chưa khoá hết")


# ----------------------------------------------------------------------------- overlap audit (bằng số, trên data thật)
def _audit_rows(store: Store, max_rows: int = 60_000, warmup: int = 2880) -> np.ndarray:
    idx = np.flatnonzero(store.eligible)
    idx = idx[idx >= warmup]  # mọi cột (lookback ≤ 2880) hữu hạn từ đây
    if len(idx) > max_rows:
        idx = idx[:: int(np.ceil(len(idx) / max_rows))]
    return idx


def _identical(a: np.ndarray, b: np.ndarray, rtol: float) -> tuple[bool, float]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.mean() < 0.5:
        return False, float("nan")
    d = float(np.max(np.abs(a[m] - b[m])))
    scale = float(max(np.std(a[m]), np.std(b[m]), 1e-12))
    return d <= rtol * scale, d / scale


def _sig(v: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(v)
    return (round(float(np.mean(v[m])), 6), round(float(np.std(v[m])), 6)) if m.any() else (np.nan, np.nan)


def _sig_close(a: tuple, b: tuple) -> bool:
    return abs(a[0] - b[0]) <= 1e-3 * max(1.0, abs(b[0])) and abs(a[1] - b[1]) <= 1e-3 * max(1e-6, b[1])


def collision_audit(store: Store, s0_by_model: dict[str, ColSet], short_cols: tuple[str, ...] = SHORT_COLUMNS,
                    ident_rtol: float = 1e-4, corr_threshold: float = 0.995, max_rows: int = 60_000,
                    dataset_label: str = "") -> dict:
    """Candidate_m = C_short \\ overlap(C_short, S0_m) cho TỪNG model; tương quan cao và trùng nội bộ C_short chỉ báo cáo.

    Trả dict máy đọc được: `c_short`, `per_model[m]` = {locked_b0/locked_ext, removed_by_overlap, candidates, near (chỉ báo)},
    `intra_short_identical` / `intra_short_near` (chỉ báo, không bỏ)."""
    idx = _audit_rows(store, max_rows)
    b0_union, ext_union = [], []
    for cs in s0_by_model.values():
        b0_union += [c for c in cs.b0 if c not in b0_union]
        ext_union += [c for c in cs.ext if c not in ext_union]
    X: dict[str, np.ndarray] = {}
    if b0_union:
        Xb = store.matrix(idx, ColSet(tuple(b0_union)))
        for i, n in enumerate(b0_union):
            X[n] = Xb[:, i].astype(np.float64)
    if ext_union:
        Xe = store.matrix(idx, ColSet((), tuple(ext_union)))
        for i, n in enumerate(ext_union):
            X[n] = Xe[:, i].astype(np.float64)
    Xs = store.matrix(idx, ColSet((), tuple(short_cols)))
    for i, n in enumerate(short_cols):
        X[n] = Xs[:, i].astype(np.float64)
    sigs = {n: _sig(v) for n, v in X.items()}
    names = list(X)
    M = np.column_stack([X[n] for n in names])
    ok = np.isfinite(M).all(axis=1)
    C = np.corrcoef(M[ok].T) if ok.sum() >= 100 else np.full((len(names), len(names)), np.nan)
    pos = {n: i for i, n in enumerate(names)}

    def near_pairs(left: list[str], right: list[str], same_set: bool) -> list[dict]:
        out = []
        for a in left:
            for b in right:
                if a == b or (same_set and pos[a] >= pos[b]):
                    continue
                c = C[pos[a], pos[b]]
                if np.isfinite(c) and abs(c) >= corr_threshold:
                    out.append({"a": a, "b": b, "corr": round(float(c), 6)})
        return out

    per_model = {}
    for m, cs in s0_by_model.items():
        s0_names = list(cs.names)
        removed, cands = [], []
        for s in short_cols:
            if s in s0_names:
                removed.append({"col": s, "reason": "trùng tên với cột S0_m", "match": s})
                continue
            hit = None
            for c in s0_names:
                if _sig_close(sigs[s], sigs[c]):
                    same, rel = _identical(X[s], X[c], ident_rtol)
                    if same:
                        hit = (c, rel)
                        break
            if hit is not None:
                removed.append({"col": s, "reason": "giá trị giống hệt một cột S0_m tại cùng timestamp", "match": hit[0],
                                "max_abs_diff_over_std": hit[1]})
            else:
                cands.append(s)
        per_model[m] = {"n_locked_b0": len(cs.b0), "n_locked_ext": len(cs.locked_ext), "locked_b0": list(cs.b0),
                        "locked_ext": list(cs.locked_ext), "removed_by_overlap": removed, "candidates": cands,
                        "n_candidates": len(cands), "near_vs_s0": near_pairs(cands, s0_names, False)}
    intra_ident = []
    for i, s in enumerate(short_cols):
        for t in short_cols[i + 1:]:
            if _sig_close(sigs[s], sigs[t]) and _identical(X[s], X[t], ident_rtol)[0]:
                intra_ident.append({"a": s, "b": t})
    return {"audit_dataset_label": dataset_label, "n_rows": int(len(idx)), "ident_rtol_vs_std": ident_rtol,
            "corr_threshold": corr_threshold, "rule": "Candidate_m = C_short \\ overlap(C_short, S0_m) per model; near/intra chỉ báo cáo",
            "short_pool_spec": short_pool_report(), "c_short": list(short_cols),
            "intra_short_identical": intra_ident, "intra_short_near": near_pairs(list(short_cols), list(short_cols), True),
            "per_model": per_model}


def candidates_from_names(names: list[str]) -> list[Candidate]:
    out = []
    for n in names:
        if n in SHORT_BY_NAME:
            out.append(SHORT_BY_NAME[n])
        else:
            raise KeyError(f"candidate không thuộc C_short: {n}")
    return out


def s0_dir(exp_dir: Path) -> Path:
    return Path(exp_dir) / "s0"


def save_lock(exp_dir: Path, s0_by_model: dict[str, ColSet], report: dict) -> None:
    d = s0_dir(exp_dir)
    d.mkdir(parents=True, exist_ok=True)
    for m, cs in s0_by_model.items():
        payload = {**cs.to_dict(), "model": m, "role": "S0_m — tập xuất phát KHOÁ toàn bộ (locked_b0 = B0*, locked_ext = F_old_m)"}
        (d / f"{m}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        pm = report["per_model"][m]
        (d / f"candidates_{m}.json").write_text(json.dumps(
            {"model": m, "rule": "Candidate_m = C_short \\ overlap(C_short, S0_m)", "s0": cs.to_dict(),
             "n_c_short": len(report["c_short"]), "candidates": pm["candidates"], "n_candidates": pm["n_candidates"],
             "removed_by_overlap": pm["removed_by_overlap"], "near_vs_s0_diagnostic_only": pm["near_vs_s0"],
             "audit_dataset_label": report["audit_dataset_label"]}, indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "collisions.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "short_pool.json").write_text(json.dumps({**report["short_pool_spec"], "intra_short_identical_diagnostic": report["intra_short_identical"]},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")


def load_lock(exp_dir: Path, model: str, dataset_label: str | None = None) -> tuple[ColSet, list[Candidate]]:
    """Đọc S0_m + Candidate_m đã khoá; kiểm schema (toàn bộ S0 khoá) và dataset của audit (hard invariant)."""
    d = s0_dir(exp_dir)
    p, q = d / f"{model}.json", d / f"candidates_{model}.json"
    if not p.exists() or not q.exists():
        raise FileNotFoundError(f"thiếu {p} / {q} — chạy `python run.py lock-s0` trước")
    raw = json.loads(p.read_text(encoding="utf-8"))
    cs = ColSet.from_dict(raw)
    assert_s0_schema(cs, raw, model)
    payload = json.loads(q.read_text(encoding="utf-8"))
    if dataset_label is not None and payload.get("audit_dataset_label") != dataset_label:
        raise ValueError(f"{q}: overlap audit chạy trên '{payload.get('audit_dataset_label')}' ≠ dataset của config "
                         f"'{dataset_label}' — chạy lại `python run.py lock-s0` trên data thật trước khi loop")
    cands = candidates_from_names(payload["candidates"])
    if set(c.name for c in cands) & set(cs.names):
        raise ValueError(f"candidates_{model}.json malformed: candidate trùng cột S0_m")
    return cs, cands
