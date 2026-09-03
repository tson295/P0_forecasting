"""C_short — candidate NGẮN HẠN (≤ 15 phút) cho vòng expanded-data (quyết định user 2026-09-03).

Đây là danh sách candidate DUY NHẤT của vòng mới: candidate cũ (39 cột §2.3, dù KEEP hay DROP) KHÔNG quay lại pool;
feature đã chọn của vòng 15 ngày nằm trong S0_m (khoá, xem `s0.py`). Không có "master feature pool" như một stage
nghiên cứu — file này chỉ là nơi định nghĩa công thức một lần cho mọi model; Candidate_m = C_short \\ overlap(C_short, S0_m).

Lưới cửa sổ W = {1, 2, 3, 4, 5, 8, 10, 15}. Mỗi họ A–O của §2.3 được làm dày NHẤT QUÁN trên lưới đó; một cửa sổ chỉ bị bỏ khi:
  (i)   suy biến về toán: EMA1-gap ≡ 0; dd_1/ru_1 ≡ 0; dd_2/ru_2 = min/max(0, r1) (r1 chỉnh lưu); σ_1 = 0; %B_2 = ±0.5;
        RSI_1 = dấu(r1); MFI_1 = dấu(ΔTP); skew cần ≥ 8 điểm; HMA cần cửa sổ ≥ 4 (WMA(k/2), WMA(√k)); rv_1 = |r1| → log(0);
  (ii)  trùng ĐỊNH NGHĨA với một cột B0-306 tại t: ret_1/ret_5/ret_8 = return1/return5/return8 của B0, log_rv5_rv60,
        rsi15_centered; hoặc là bản affine/xấp xỉ của cột B0 (log_range_1 ≈ log(H/L), ad_vwclv_1 = 2·close_position_t,
        ATR15 ≈ ATR14 cũ, HMA15 ≈ HMA16 của B0) — kiểm tra bằng số trong `s0.collision_audit`, không chỉ bằng suy luận;
  (iii) cửa sổ đó đã là candidate của vòng 15 ngày (vwap 1/15, ret 60+, bb 20/60, atr 14, kcw 20, mfi 14/60, ad 5/15/60,
        dd/ru 240, range 240, skew 60, hma 64, rsi 240, macd 60/240/60).
Mọi công thức causal: cửa sổ kết thúc tại t, chỉ dùng bar ≤ t. Ký hiệu như `features_ext` (A = amount, TP = (H+L+C)/3,
r1 = Δlog C, rv_k = sqrt(mean_k r1²), EMA_k = ewm(span k, min_periods k) trên log C — helper của B0, reset sau gap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from Baseline_LGBM import _ema, _hma, _rsi

from .features_ext import Candidate, _atr, _rsum, _rv, _safe_log

SHORT_GRID = (1, 2, 3, 4, 5, 8, 10, 15)

# Họ → (cửa sổ dùng, {cửa sổ bỏ: lý do}) — bảng này là nguồn sự thật cho báo cáo `s0/short_pool.json`.
SHORT_FAMILIES: dict[str, dict] = {
    "A_vwap_amt_gap": {"use": (2, 3, 4, 5, 8, 10), "skip": {1: "candidate cũ vwap_amt_gap_1", 15: "candidate cũ vwap_amt_gap_15"}},
    "B_ret": {"use": (2, 3, 4, 10, 15), "skip": {1: "= B0 fine:t:return1", 5: "= B0 fine:t:return5", 8: "= B0 coarse:t:return8"}},
    "C_log_rv_rv60": {"use": (2, 3, 4, 8, 10, 15), "skip": {1: "rv_1 = |r1| → log(0) khi r1 = 0", 5: "= B0 fine:t:log_rv5_rv60"}},
    "D_log_c_ema": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "EMA_1 = log C → gap ≡ 0"}},
    "D_log_ema_ema": {"use": ((2, 8), (3, 10), (5, 15)), "skip": {}},
    "E_rsi": {"use": (2, 3, 4, 5, 8, 10), "skip": {1: "RSI_1 = dấu(r1) suy biến", 15: "= B0 fine:t:rsi15_centered"}},
    "F_bb_pctb": {"use": (3, 4, 5, 8, 10, 15), "skip": {1: "σ_1 = 0", 2: "%B_2 = ±0.5 suy biến"}},
    "F_bb_logbw": {"use": (3, 4, 5, 8, 10, 15), "skip": {1: "σ_1 = 0", 2: "σ_2 = |Δlog C|/2 = |r1|/2 (bản affine của |r1|)"}},
    "G_log_atr_c": {"use": (2, 3, 4, 5, 8, 10), "skip": {1: "ATR_1 = TR ≈ H−L → ≈ B0 log_range", 15: "≈ ATR14 (candidate cũ log_atr14_c)"}},
    "G_log_atr_rv": {"use": (2, 3, 4, 5, 8, 10), "skip": {1: "rv_1 = |r1| → log(0)", 15: "≈ ATR14/rv14 (candidate cũ log_atr14_rv14)"}},
    # Keltner width ngắn: kcw_k = log 2 + log(ATR_k/C) + log(C/EMA_k(C)); số hạng cuối ≈ 0 ở k ≤ 10 → trùng số học với
    # log_atr{k}_c (collision audit trên data 15 ngày thật 2026-09-03: corr = 1.000000 ở k = 2–5, 0.999999 ở k = 8, 10)
    # → suy biến, KHÔNG sinh (kcw_20 cũ vẫn nằm trong S0 của model đã KEEP nó).
    "H_kcw": {"use": (), "skip": {k: "≡ log_atr{k}_c + const (corr ≥ 0.999999 trên data thật) → suy biến" for k in SHORT_GRID}},
    "I_mfi": {"use": (2, 3, 4, 5, 8, 10), "skip": {1: "MFI_1 ∈ {−0.5, +0.5, NaN} = dấu(ΔTP)", 15: "≈ MFI14 (candidate cũ mfi14_centered)"}},
    "J_ad_vwclv": {"use": (2, 3, 4, 8, 10), "skip": {1: "= CLV_t = 2·close_position_t của B0 (affine)", 5: "candidate cũ", 15: "candidate cũ"}},
    "K_dd": {"use": (3, 4, 5, 8, 10, 15), "skip": {1: "dd_1 ≡ 0", 2: "dd_2 = min(0, r1) (r1 chỉnh lưu)"}},
    "K_ru": {"use": (3, 4, 5, 8, 10, 15), "skip": {1: "ru_1 ≡ 0", 2: "ru_2 = max(0, r1) (r1 chỉnh lưu)"}},
    "L_log_range": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "log((H−L)/C) ≈ B0 fine:t:log_range = log(H/L)"}},
    "M_ret_skew": {"use": (8, 10, 15), "skip": {k: "skew cần ≥ 8 điểm để có nghĩa thống kê" for k in (1, 2, 3, 4, 5)}},
    "N_hma_slope": {"use": (4, 5, 8, 10), "skip": {1: "HMA cần k ≥ 4", 2: "HMA cần k ≥ 4", 3: "HMA cần k ≥ 4", 15: "≈ B0 hma_slope16_volnorm"}},
    "O_macd_hist": {"use": ((2, 8, 3), (3, 10, 3), (5, 15, 5)), "skip": {}},
}


def _build() -> list[Candidate]:
    out: list[Candidate] = []

    def add(name: str, group: str) -> None:
        out.append(Candidate(name, (name,), group))

    for k in SHORT_FAMILIES["A_vwap_amt_gap"]["use"]:
        add(f"vwap_amt_gap_{k}", "A")
    for k in SHORT_FAMILIES["B_ret"]["use"]:
        add(f"ret_{k}", "B")
    for k in SHORT_FAMILIES["C_log_rv_rv60"]["use"]:
        add(f"log_rv{k}_rv60", "C")
    for k in SHORT_FAMILIES["D_log_c_ema"]["use"]:
        add(f"log_c_ema{k}", "D")
    for a, b in SHORT_FAMILIES["D_log_ema_ema"]["use"]:
        add(f"log_ema{a}_ema{b}", "D")
    for k in SHORT_FAMILIES["E_rsi"]["use"]:
        add(f"rsi{k}_centered", "E")
    for k in SHORT_FAMILIES["F_bb_pctb"]["use"]:
        add(f"bb_pctb_{k}", "F")
    for k in SHORT_FAMILIES["F_bb_logbw"]["use"]:
        add(f"bb_logbw_{k}", "F")
    for k in SHORT_FAMILIES["G_log_atr_c"]["use"]:
        add(f"log_atr{k}_c", "G")
    for k in SHORT_FAMILIES["G_log_atr_rv"]["use"]:
        add(f"log_atr{k}_rv{k}", "G")
    for k in SHORT_FAMILIES["H_kcw"]["use"]:
        add(f"kcw_{k}", "H")
    for k in SHORT_FAMILIES["I_mfi"]["use"]:
        add(f"mfi{k}_centered", "I")
    for k in SHORT_FAMILIES["J_ad_vwclv"]["use"]:
        add(f"ad_vwclv_{k}", "J")
    for k in SHORT_FAMILIES["K_dd"]["use"]:
        add(f"dd_{k}", "K")
    for k in SHORT_FAMILIES["K_ru"]["use"]:
        add(f"ru_{k}", "K")
    for k in SHORT_FAMILIES["L_log_range"]["use"]:
        add(f"log_range_{k}", "L")
    for k in SHORT_FAMILIES["M_ret_skew"]["use"]:
        add(f"ret_skew_{k}", "M")
    for k in SHORT_FAMILIES["N_hma_slope"]["use"]:
        add(f"hma_slope{k}_volnorm", "N")
    for f, s, sig in SHORT_FAMILIES["O_macd_hist"]["use"]:
        add(f"macd_hist_{f}_{s}_{sig}_volnorm", "O")
    return out


SHORT_CANDIDATES: list[Candidate] = _build()
SHORT_BY_NAME = {c.name: c for c in SHORT_CANDIDATES}
SHORT_COLUMNS = tuple(c.name for c in SHORT_CANDIDATES)


def _parse(name: str) -> tuple[str, tuple[int, ...]]:
    """Tên cột → (họ, tham số cửa sổ)."""
    import re

    for pat, fam in (
        (r"^vwap_amt_gap_(\d+)$", "A"), (r"^ret_(\d+)$", "B"), (r"^log_rv(\d+)_rv60$", "C"), (r"^log_c_ema(\d+)$", "D"),
        (r"^log_ema(\d+)_ema(\d+)$", "D2"), (r"^rsi(\d+)_centered$", "E"), (r"^bb_pctb_(\d+)$", "F"), (r"^bb_logbw_(\d+)$", "F2"),
        (r"^log_atr(\d+)_c$", "G"), (r"^log_atr(\d+)_rv(\d+)$", "G2"), (r"^kcw_(\d+)$", "H"), (r"^mfi(\d+)_centered$", "I"),
        (r"^ad_vwclv_(\d+)$", "J"), (r"^dd_(\d+)$", "K"), (r"^ru_(\d+)$", "K2"), (r"^log_range_(\d+)$", "L"),
        (r"^ret_skew_(\d+)$", "M"), (r"^hma_slope(\d+)_volnorm$", "N"), (r"^macd_hist_(\d+)_(\d+)_(\d+)_volnorm$", "O"),
    ):
        m = re.match(pat, name)
        if m:
            return fam, tuple(int(x) for x in m.groups())
    raise KeyError(f"không phải cột C_short: {name}")


def compute_short(g: pd.DataFrame, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Tính các cột C_short trên lưới g (index UTC, cột lowercase open/high/low/close/volume/amount; NaN tại gap).

    Mọi cửa sổ kết thúc tại t; không dùng bar > t. Cùng quy ước NaN ở warmup như `features_ext.compute_ext`."""
    want = tuple(columns) if columns is not None else SHORT_COLUMNS
    c, o, h, l, v, a = (g[k].astype(float) for k in ("close", "open", "high", "low", "volume", "amount"))
    logc = _safe_log(c)
    r1 = logc.diff()
    rv60 = _rv(r1, 60)
    rv60p = rv60.where(rv60 > 0)
    tp = (h + l + c) / 3.0
    out: dict[str, pd.Series] = {}
    cache: dict = {}

    def ema(k: int) -> pd.Series:
        if ("ema", k) not in cache:
            cache[("ema", k)] = _ema(logc, k)
        return cache[("ema", k)]

    def atr(k: int) -> pd.Series:
        if ("atr", k) not in cache:
            cache[("atr", k)] = _atr(h, l, c, k)
        return cache[("atr", k)]

    for name in want:
        fam, p = _parse(name)
        k = p[0]
        if fam == "A":
            vw = _rsum(a, k) / _rsum(v, k).where(_rsum(v, k) > 0)
            out[name] = _safe_log(c / vw.where(vw > 0))
        elif fam == "B":
            out[name] = logc.diff(k)
        elif fam == "C":
            out[name] = _safe_log(_rv(r1, k) / rv60p)
        elif fam == "D":
            out[name] = logc - ema(k)
        elif fam == "D2":
            out[name] = ema(p[0]) - ema(p[1])
        elif fam == "E":
            out[name] = _rsi(r1, k) / 100.0 - 0.5
        elif fam in ("F", "F2"):
            sma = logc.rolling(k, min_periods=k).mean()
            sd = logc.rolling(k, min_periods=k).std(ddof=0)
            out[name] = (logc - sma) / (2.0 * sd.where(sd > 0)) if fam == "F" else _safe_log(sd)
        elif fam == "G":
            out[name] = _safe_log(atr(k) / c)
        elif fam == "G2":
            rvk = _rv(r1, p[1])
            out[name] = _safe_log((atr(k) / c) / rvk.where(rvk > 0))
        elif fam == "H":
            emac = c.ewm(span=k, adjust=False, min_periods=k).mean()  # cùng định nghĩa với kcw_20 của §2.3
            out[name] = _safe_log(2.0 * atr(k) / emac.where(emac > 0))
        elif fam == "I":
            dtp = tp.diff()
            a_pos = a.where(dtp > 0, 0.0).where(dtp.notna())
            a_neg = a.where(dtp < 0, 0.0).where(dtp.notna())
            pos, neg = _rsum(a_pos, k), _rsum(a_neg, k)
            den = pos + neg
            out[name] = pos / den.where(den > 0) - 0.5
        elif fam == "J":
            rng_ = h - l
            clv = (((c - l) - (h - c)) / rng_.where(rng_ > 0)).fillna(0.0).where(rng_.notna())
            out[name] = _rsum(clv * v, k) / _rsum(v, k).where(_rsum(v, k) > 0)
        elif fam == "K":
            out[name] = _safe_log(c / c.rolling(k, min_periods=k).max())
        elif fam == "K2":
            out[name] = _safe_log(c / c.rolling(k, min_periods=k).min())
        elif fam == "L":
            out[name] = _safe_log((h.rolling(k, min_periods=k).max() - l.rolling(k, min_periods=k).min()) / c)
        elif fam == "M":
            out[name] = r1.rolling(k, min_periods=k).skew()
        elif fam == "N":
            out[name] = _hma(logc, k).diff() / rv60p
        elif fam == "O":
            f, s, sig = p
            macd = ema(f) - ema(s)
            out[name] = (macd - _ema(macd, sig)) / rv60p
    df = pd.DataFrame(out, index=g.index).replace([np.inf, -np.inf], np.nan)
    return df[list(want)].astype(np.float32)


def short_pool_report() -> dict:
    """Bảng máy đọc được: họ → cửa sổ dùng / cửa sổ bỏ + lý do; tổng số cột."""
    return {"grid": list(SHORT_GRID), "n_candidates": len(SHORT_CANDIDATES),
            "families": {fam: {"use": [list(u) if isinstance(u, tuple) else u for u in spec["use"]],
                               "skip": {str(k): v for k, v in spec["skip"].items()}} for fam, spec in SHORT_FAMILIES.items()},
            "columns": list(SHORT_COLUMNS)}
