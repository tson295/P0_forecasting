"""S0_m KHOÁ + kiểm tra trùng (collision audit) + Candidate_m — vòng expanded-data (quyết định user 2026-09-03).

- S0_m = B0* ∪ F_old_m, dựng từ ARTIFACT thắng của vòng 15 ngày (`<prev_run_dir>/wins/<m>.json`, `b0_star.json`),
  không gõ tay. `locked` = toàn bộ F_old_m: không phải candidate, không bị prune PI bỏ, không thể `without_ext`.
  AutoTS: mỗi nhánh probe (autots_wr / autots_mr) kế thừa ĐÚNG bộ thắng của nhánh đó (không dùng AutoTS-final).
  TimesFM: TimesFM-final cũ = native (0 covariate) → S0_TFM = ∅, không bịa covariate kế thừa (§7 quyết định).
- Collision = trùng tên CHÍNH XÁC, hoặc khác tên nhưng giá trị GIỐNG HỆT tại cùng timestamp (kiểm bằng số trên data thật,
  `ident_rtol` so với std cột). Cùng họ chỉ báo, tương quan cao chỉ báo (`near`, KHÔNG tự xoá); cùng indicator khác lag
  KHÔNG phải trùng (khác timestamp) — ví dụ `fine:t-4m:rsi15_centered` ≠ rsi15 tại t.
- Candidate_m = C_short \\ overlap(C_short, S0_m): overlap gồm trùng tên và trùng giá trị. C_short đã loại sẵn mọi cột
  §2.3 cũ (KEEP lẫn DROP) theo cách xây lưới (`features_short`), nên không có gì để "trừ DROP cũ"; ngoài ra cột C_short
  trùng giá trị với BẤT KỲ cột B0-306 hoặc candidate cũ nào bị loại khỏi pool (không phải feature mới) và được ghi lý do.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features_ext import ALL_EXT_COLUMNS, Candidate
from .features_short import SHORT_BY_NAME, SHORT_CANDIDATES, SHORT_COLUMNS, short_pool_report
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
    """Cột DROP của vòng trước (keepdrop_<m>.csv) — chỉ để kiểm tra chúng KHÔNG quay lại pool."""
    p = Path(prev_dir) / f"keepdrop_{model}.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    return [c for cols in d.loc[d["decision"] == "DROP", "columns"] for c in str(cols).split("|")]


def s0_for(model: str, prev_dir: Path | None, b0_star: ColSet | None = None) -> ColSet:
    """S0_m: từ winner vòng trước (locked = F_old_m); TimesFM = ∅; không có vòng trước (smoke) → B0*."""
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


# ----------------------------------------------------------------------------- collision audit (bằng số, trên data thật)
def _audit_rows(store: Store, max_rows: int = 60_000, warmup: int = 2880) -> np.ndarray:
    idx = np.flatnonzero(store.eligible)
    idx = idx[idx >= warmup]  # mọi cột §2.3 cũ (lookback ≤ 2880) và C_short đều hữu hạn từ đây
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


def collision_audit(store: Store, s0_by_model: dict[str, ColSet], short_cols: tuple[str, ...] = SHORT_COLUMNS,
                    ident_rtol: float = 1e-4, corr_threshold: float = 0.995, max_rows: int = 60_000,
                    dataset_label: str = "") -> dict:
    """Kiểm tra trùng giữa C_short và (B0-306 ∪ candidate cũ §2.3 ∪ chính C_short), rồi suy ra Candidate_m cho từng S0_m.

    Trả dict máy đọc được: `identical` (bị loại), `near` (chỉ báo), `pool` (C_short sau loại), `per_model` (S0 + Candidate_m).
    """
    idx = _audit_rows(store, max_rows)
    b0 = store.all_b0()
    X = {}
    Xb = store.matrix(idx, b0)
    for i, n in enumerate(b0.names):
        X[n] = Xb[:, i].astype(np.float64)
    Xo = store.matrix(idx, ColSet((), ALL_EXT_COLUMNS))
    for i, n in enumerate(ALL_EXT_COLUMNS):
        X[n] = Xo[:, i].astype(np.float64)
    Xs = store.matrix(idx, ColSet((), tuple(short_cols)))
    for i, n in enumerate(short_cols):
        X[n] = Xs[:, i].astype(np.float64)
    reference = list(b0.names) + list(ALL_EXT_COLUMNS)
    # prefilter theo chữ ký (mean, std) làm tròn để không so 100 × 350 cặp đầy đủ
    def sig(v):
        m = np.isfinite(v)
        return (round(float(np.mean(v[m])), 6), round(float(np.std(v[m])), 6)) if m.any() else (np.nan, np.nan)

    sigs = {n: sig(v) for n, v in X.items()}
    identical, excluded, pool = [], {}, []
    for j, s in enumerate(short_cols):
        hit = None
        for other in reference + list(pool):
            if other == s:
                continue
            ms, mo = sigs[s], sigs[other]
            if not (abs(ms[0] - mo[0]) <= 1e-3 * max(1.0, abs(mo[0])) and abs(ms[1] - mo[1]) <= 1e-3 * max(1e-6, mo[1])):
                continue
            same, rel = _identical(X[s], X[other], ident_rtol)
            if same:
                hit = (other, rel)
                break
        if hit is not None:
            kind = "B0-306" if hit[0] in b0.names else ("candidate cũ §2.3" if hit[0] in ALL_EXT_COLUMNS else "C_short (trước nó)")
            identical.append({"short": s, "match": hit[0], "kind": kind, "max_abs_diff_over_std": hit[1]})
            excluded[s] = f"trùng giá trị với {kind}: {hit[0]}"
        else:
            pool.append(s)
    # near-duplicate (chỉ báo): |corr| ≥ ngưỡng trên hàng mọi cột hữu hạn
    names = list(pool) + reference
    M = np.column_stack([X[n] for n in names])
    ok = np.isfinite(M).all(axis=1)
    near = []
    if ok.sum() >= 100:
        C = np.corrcoef(M[ok].T)
        for a in range(len(pool)):
            for b in range(a + 1, len(names)):
                c = C[a, b]
                if np.isfinite(c) and abs(c) >= corr_threshold:
                    near.append({"a": names[a], "b": names[b], "corr": round(float(c), 6),
                                 "kind": "B0-306" if names[b] in b0.names else ("candidate cũ §2.3" if names[b] in ALL_EXT_COLUMNS else "C_short")})
    # per model: Candidate_m = pool \ overlap(pool, S0_m) — trùng tên hoặc trùng giá trị với cột của S0_m
    per_model = {}
    for m, cs in s0_by_model.items():
        s0_names = set(cs.names)
        removed, cands = [], []
        for s in pool:
            if s in s0_names:
                removed.append({"col": s, "reason": "trùng tên với cột S0"})
                continue
            dup = next((r for r in identical if r["short"] == s and r["match"] in s0_names), None)
            if dup is not None:
                removed.append({"col": s, "reason": f"trùng giá trị với {dup['match']}"})
                continue
            cands.append(s)
        per_model[m] = {"n_b0": len(cs.b0), "n_locked_ext": len(cs.locked), "locked_ext": list(cs.locked),
                        "removed_by_overlap": removed, "candidates": cands, "n_candidates": len(cands)}
    return {"audit_dataset_label": dataset_label, "n_rows": int(len(idx)), "ident_rtol_vs_std": ident_rtol,
            "corr_threshold": corr_threshold, "short_pool_spec": short_pool_report(),
            "identical": identical, "excluded_from_pool": excluded, "near": near, "pool": pool,
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
        cs.save(d / f"{m}.json")
        pm = report["per_model"][m]
        (d / f"candidates_{m}.json").write_text(json.dumps(
            {"model": m, "s0": cs.to_dict(), "candidates": pm["candidates"], "removed_by_overlap": pm["removed_by_overlap"],
             "audit_dataset_label": report["audit_dataset_label"]}, indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "collisions.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "short_pool.json").write_text(json.dumps({**report["short_pool_spec"], "pool_after_exclusion": report["pool"],
                                                    "excluded": report["excluded_from_pool"]}, indent=1, ensure_ascii=False),
                                       encoding="utf-8")


def load_lock(exp_dir: Path, model: str, dataset_label: str | None = None) -> tuple[ColSet, list[Candidate]]:
    """Đọc S0_m + Candidate_m đã khoá. `dataset_label` ≠ None → collision audit phải được chạy trên ĐÚNG dataset đó
    (audit trên data khác chỉ để xem trước; trước khi `loop` trên data thật phải chạy lại `lock-s0` không có --data-config)."""
    d = s0_dir(exp_dir)
    p, q = d / f"{model}.json", d / f"candidates_{model}.json"
    if not p.exists() or not q.exists():
        raise FileNotFoundError(f"thiếu {p} / {q} — chạy `python run.py lock-s0` trước")
    cs = ColSet.load(p)
    payload = json.loads(q.read_text(encoding="utf-8"))
    if dataset_label is not None and payload.get("audit_dataset_label") != dataset_label:
        raise ValueError(f"{q}: collision audit chạy trên '{payload.get('audit_dataset_label')}' ≠ dataset của config "
                         f"'{dataset_label}' — chạy lại `python run.py lock-s0` trên data thật trước khi loop")
    cands = candidates_from_names(payload["candidates"])
    return cs, cands
