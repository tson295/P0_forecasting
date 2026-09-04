"""C_short — candidate NGẮN HẠN (≤ 15 phút) cho vòng expanded-data (quyết định user 2026-09-03, hiệu chỉnh 2026-09-04).

C_short là tập ĐỊNH NGHĨA candidate mới, sinh MỘT LẦN cho mọi model (tiện thực thi, KHÔNG phải stage nghiên cứu).
Việc lọc duy nhất diễn ra theo từng model: `Candidate_m = C_short \\ overlap(C_short, S0_m)` (xem `s0.py`) — chỉ bỏ cột
đã có trong S0_m của CHÍNH model đó (trùng tên, hoặc khác tên nhưng giá trị giống hệt tại cùng timestamp). KHÔNG lọc toàn cục
theo B0-306, KHÔNG lọc theo tương quan cao (tương quan chỉ là chẩn đoán, báo cáo), KHÔNG bỏ vì "xấp xỉ" một cột khác.

Lưới cửa sổ W = {1, 2, 3, 4, 5, 8, 10, 15}: mọi họ được làm DÀY trên lưới đó; một cửa sổ chỉ bị bỏ khi
  (i)  suy biến/không xác định về TOÁN (ví dụ log C − EMA_1 ≡ 0, dd_1 ≡ 0, σ_1 = 0 → chia 0, rv_1 = |r1| → log 0 khi r1 = 0,
       skew cần ≥ 3 quan sát, PSAR cần ≥ 2 bar, kcw_1 = log 2 + log_atr1_c là đồng nhất thức);
  (ii) cửa sổ đó CHÍNH LÀ một candidate cũ §2.3 (vòng 15 ngày — KEEP hay DROP đều không quay lại, §6 quyết định; ví dụ
       vwap_amt_gap_1/15, ad_vwclv_5/15, r5_1, log_c5_ema5_12).
Mọi lý do bỏ đều ghi máy đọc được trong `SHORT_FAMILIES[...]["skip"]` → `s0/short_pool.json`.
Ngoại lệ có tài liệu: `dow` (thứ trong tuần) KHÔNG có cửa sổ tự nhiên → không sinh biến thể (họ T).
Mọi công thức causal: cửa sổ kết thúc tại t, chỉ dùng bar ≤ t; feature 5 phút dùng bar 5' ĐÃ ĐÓNG (as-of T ≤ t).
Ký hiệu như `features_ext` (A = amount, TP = (H+L+C)/3, r1 = Δlog C, rv_k = sqrt(mean_k r1²), EMA_k = ewm(span k) — helper của B0).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from Baseline_LGBM import _ema, _hma, _rsi

from .data import asof_index
from .features_ext import Candidate, _atr, _rsum, _rv, _safe_log

SHORT_GRID = (1, 2, 3, 4, 5, 8, 10, 15)
_RV1 = "rv_1 = |r1| → log(0) = −∞ khi r1 = 0 (không xác định)"
_OLD = "chính là candidate cũ §2.3 ({}) — không quay lại (§6)"

# Họ → (cửa sổ dùng, {cửa sổ bỏ: lý do CHÍNH XÁC}) — nguồn sự thật cho `s0/short_pool.json`.
SHORT_FAMILIES: dict[str, dict] = {
    "A_vwap_amt_gap": {"use": (2, 3, 4, 5, 8, 10), "skip": {1: _OLD.format("vwap_amt_gap_1"), 15: _OLD.format("vwap_amt_gap_15")}},
    "B_ret": {"use": SHORT_GRID, "skip": {}},
    "C_log_rv_rv60": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: _RV1}},
    "D_log_c_ema": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "log C − EMA_1(log C) ≡ 0"}},
    "D_log_ema_ema": {"use": ((2, 5), (2, 8), (3, 10), (4, 10), (5, 15)), "skip": {}},
    "E_rsi": {"use": SHORT_GRID, "skip": {}},
    "F_bb_pctb": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "σ_1 = 0 → chia 0 (không xác định)"}},
    "F_bb_logbw": {"use": (3, 4, 5, 8, 10, 15), "skip": {1: "σ_1 = 0 → log 0", 2: "σ_2 = |Δlog C|/2 → log 0 khi r1 = 0 (không xác định)"}},
    "G_log_atr_c": {"use": SHORT_GRID, "skip": {}},
    "G_log_atr_rv": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: _RV1}},
    "H_kcw": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "kcw_1 = log(2·TR/C) = log 2 + log_atr1_c (đồng nhất thức, chỉ lệch hằng số)"}},
    "I_mfi": {"use": SHORT_GRID, "skip": {}},
    "J_ad_vwclv": {"use": (1, 2, 3, 4, 8, 10), "skip": {5: _OLD.format("ad_vwclv_5"), 15: _OLD.format("ad_vwclv_15")}},
    "K_dd": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "dd_1 = log(C / max_1 C) ≡ 0"}},
    "K_ru": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "ru_1 = log(C / min_1 C) ≡ 0"}},
    "L_log_range": {"use": SHORT_GRID, "skip": {}},
    "M_ret_skew": {"use": (3, 4, 5, 8, 10, 15), "skip": {1: "skew cần ≥ 3 quan sát (không xác định)", 2: "skew cần ≥ 3 quan sát (không xác định)"}},
    "N_hma_slope": {"use": SHORT_GRID, "skip": {}},
    "O_macd_hist": {"use": ((2, 5, 2), (2, 8, 3), (3, 10, 3), (4, 10, 4), (5, 15, 5)), "skip": {}},
    # PSAR cửa sổ reset: khởi tạo lại trạng thái Wilder PSAR tại đầu W bar cuối [t−W+1 … t], chạy causal, lấy trạng thái tại t.
    "P_psar_dir": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "PSAR cần ≥ 2 bar để khởi tạo hướng/EP (n < 2 → trạng thái không xác định)"}},
    "P_psar_logdist": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "PSAR cần ≥ 2 bar để khởi tạo hướng/EP (n < 2 → trạng thái không xác định)"}},
    "P_psar_age_log": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: "PSAR cần ≥ 2 bar để khởi tạo hướng/EP (n < 2 → trạng thái không xác định)"}},
    "Q_log_rv_med2d": {"use": (2, 3, 4, 5, 8, 10, 15), "skip": {1: _RV1}},
    # Họ 5 phút: cửa sổ tính bằng bar 5' đã đóng (k bar = 5k phút ≤ 15 → k ∈ {1, 2, 3}); k = 1 và 12 là candidate cũ.
    "R_r5": {"use": (2, 3), "skip": {1: _OLD.format("r5_1"), 12: _OLD.format("r5_12")}},
    "S_log_c5_ema5": {"use": (2, 3), "skip": {1: "log C5 − EMA_1(log C5) ≡ 0", 12: _OLD.format("log_c5_ema5_12")}},
    # DOW: NGOẠI LỆ có tài liệu — thứ trong tuần không có cửa sổ cuộn tự nhiên 2'/5'/15'; không sinh biến thể nào.
    "T_dow": {"use": (), "skip": {"*": "ngoại lệ: weekday không có horizon cuộn tự nhiên; dow_sin/dow_cos là candidate cũ §2.3 (§5C)"}},
}


def _build() -> list[Candidate]:
    out: list[Candidate] = []

    def add(name: str, group: str) -> None:
        out.append(Candidate(name, (name,), group))

    F = SHORT_FAMILIES
    for k in F["A_vwap_amt_gap"]["use"]:
        add(f"vwap_amt_gap_{k}", "A")
    for k in F["B_ret"]["use"]:
        add(f"ret_{k}", "B")
    for k in F["C_log_rv_rv60"]["use"]:
        add(f"log_rv{k}_rv60", "C")
    for k in F["D_log_c_ema"]["use"]:
        add(f"log_c_ema{k}", "D")
    for a, b in F["D_log_ema_ema"]["use"]:
        add(f"log_ema{a}_ema{b}", "D")
    for k in F["E_rsi"]["use"]:
        add(f"rsi{k}_centered", "E")
    for k in F["F_bb_pctb"]["use"]:
        add(f"bb_pctb_{k}", "F")
    for k in F["F_bb_logbw"]["use"]:
        add(f"bb_logbw_{k}", "F")
    for k in F["G_log_atr_c"]["use"]:
        add(f"log_atr{k}_c", "G")
    for k in F["G_log_atr_rv"]["use"]:
        add(f"log_atr{k}_rv{k}", "G")
    for k in F["H_kcw"]["use"]:
        add(f"kcw_{k}", "H")
    for k in F["I_mfi"]["use"]:
        add(f"mfi{k}_centered", "I")
    for k in F["J_ad_vwclv"]["use"]:
        add(f"ad_vwclv_{k}", "J")
    for k in F["K_dd"]["use"]:
        add(f"dd_{k}", "K")
    for k in F["K_ru"]["use"]:
        add(f"ru_{k}", "K")
    for k in F["L_log_range"]["use"]:
        add(f"log_range_{k}", "L")
    for k in F["M_ret_skew"]["use"]:
        add(f"ret_skew_{k}", "M")
    for k in F["N_hma_slope"]["use"]:
        add(f"hma_slope{k}_volnorm", "N")
    for f, s, sig in F["O_macd_hist"]["use"]:
        add(f"macd_hist_{f}_{s}_{sig}_volnorm", "O")
    for k in F["P_psar_dir"]["use"]:
        add(f"psar_dir_{k}", "P")
    for k in F["P_psar_logdist"]["use"]:
        add(f"psar_logdist_{k}", "P")
    for k in F["P_psar_age_log"]["use"]:
        add(f"psar_age_log_{k}", "P")
    for k in F["Q_log_rv_med2d"]["use"]:
        add(f"log_rv{k}_med2d", "Q")
    for k in F["R_r5"]["use"]:
        add(f"r5_{k}", "R")
    for k in F["S_log_c5_ema5"]["use"]:
        add(f"log_c5_ema5_{k}", "S")
    return out


SHORT_CANDIDATES: list[Candidate] = _build()
SHORT_BY_NAME = {c.name: c for c in SHORT_CANDIDATES}
SHORT_COLUMNS = tuple(c.name for c in SHORT_CANDIDATES)

_PATTERNS = (
    (r"^vwap_amt_gap_(\d+)$", "A"), (r"^ret_(\d+)$", "B"), (r"^log_rv(\d+)_rv60$", "C"), (r"^log_c_ema(\d+)$", "D"),
    (r"^log_ema(\d+)_ema(\d+)$", "D2"), (r"^rsi(\d+)_centered$", "E"), (r"^bb_pctb_(\d+)$", "F"), (r"^bb_logbw_(\d+)$", "F2"),
    (r"^log_atr(\d+)_c$", "G"), (r"^log_atr(\d+)_rv(\d+)$", "G2"), (r"^kcw_(\d+)$", "H"), (r"^mfi(\d+)_centered$", "I"),
    (r"^ad_vwclv_(\d+)$", "J"), (r"^dd_(\d+)$", "K"), (r"^ru_(\d+)$", "K2"), (r"^log_range_(\d+)$", "L"),
    (r"^ret_skew_(\d+)$", "M"), (r"^hma_slope(\d+)_volnorm$", "N"), (r"^macd_hist_(\d+)_(\d+)_(\d+)_volnorm$", "O"),
    (r"^psar_dir_(\d+)$", "P1"), (r"^psar_logdist_(\d+)$", "P2"), (r"^psar_age_log_(\d+)$", "P3"),
    (r"^log_rv(\d+)_med2d$", "Q"), (r"^r5_(\d+)$", "R"), (r"^log_c5_ema5_(\d+)$", "S"),
)


def _parse(name: str) -> tuple[str, tuple[int, ...]]:
    """Tên cột → (họ, tham số cửa sổ)."""
    for pat, fam in _PATTERNS:
        m = re.match(pat, name)
        if m:
            return fam, tuple(int(x) for x in m.groups())
    raise KeyError(f"không phải cột C_short: {name}")


# ----------------------------------------------------------------------------- PSAR cửa sổ reset (vector hoá)
def psar_window(high: np.ndarray, low: np.ndarray, close: np.ndarray, W: int, af0: float = 0.02, af_step: float = 0.02,
                af_max: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PSAR Wilder khởi tạo lại tại đầu cửa sổ W bar [t−W+1 … t], chạy causal, trả trạng thái tại t: (dir, sar, age).

    Cùng quy tắc với `features_ext._psar_segment` (khởi tạo hướng từ 2 bar đầu, flip, EP/AF, clamp theo 2 bar gần nhất) nhưng
    tính đồng thời cho mọi t bằng cửa sổ trượt; cửa sổ chứa NaN (gap) → NaN. W < 2 → không có trạng thái (NaN)."""
    n = len(close)
    d_out, s_out, a_out = (np.full(n, np.nan) for _ in range(3))
    if W < 2 or n < W:
        return d_out, s_out, a_out
    H, L, C = (sliding_window_view(np.asarray(x, float), W) for x in (high, low, close))
    valid = np.isfinite(H).all(axis=1) & np.isfinite(L).all(axis=1) & np.isfinite(C).all(axis=1)
    with np.errstate(invalid="ignore"):
        up = C[:, 1] >= C[:, 0]
        sar = np.where(up, L[:, 0], H[:, 0])
        ep = np.where(up, H[:, 0], L[:, 0])
        af = np.full(len(up), af0)
        since = np.zeros(len(up))
        for i in range(1, W):
            hi, lo = H[:, i], L[:, i]
            flip = np.where(up, lo < sar, hi > sar)
            better = np.where(up, hi > ep, lo < ep)
            ep_nf = np.where(better, np.where(up, hi, lo), ep)
            af_nf = np.where(better, np.minimum(af + af_step, af_max), af)
            new_up = np.where(flip, ~up, up)
            sar = np.where(flip, ep, sar)
            ep = np.where(flip, np.where(up, lo, hi), ep_nf)
            af = np.where(flip, af0, af_nf)
            since = np.where(flip, 0.0, since + 1.0)
            up = new_up
            nxt = sar + af * (ep - sar)
            sar = np.where(up, np.minimum(nxt, np.minimum(lo, L[:, i - 1])), np.maximum(nxt, np.maximum(hi, H[:, i - 1])))
    end = np.arange(W - 1, n)
    d_out[end] = np.where(valid, np.where(up, 1.0, -1.0), np.nan)
    s_out[end] = np.where(valid, sar, np.nan)
    a_out[end] = np.where(valid, since, np.nan)
    return d_out, s_out, a_out


def compute_short(g: pd.DataFrame, lf: pd.DataFrame | None = None, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Tính các cột C_short trên lưới g (index UTC, cột lowercase open/high/low/close/volume/amount; NaN tại gap).

    `lf` = bar 5 phút (họ R/S; thiếu → NaN như `features_ext`). Mọi cửa sổ kết thúc tại t; không dùng bar > t."""
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

    def psar(W: int):
        if ("psar", W) not in cache:
            cache[("psar", W)] = psar_window(h.to_numpy(float), l.to_numpy(float), c.to_numpy(float), W)
        return cache[("psar", W)]

    def lf5():
        if "lf5" not in cache:
            if lf is None:
                cache["lf5"] = None
            else:
                lfs = lf.sort_values("timestamp").reset_index(drop=True)
                logc5 = _safe_log(lfs["close"].astype(float))
                idx = asof_index(lfs["timestamp"].to_numpy(np.int64), g["timestamp"].to_numpy(np.int64))
                cache["lf5"] = (logc5, idx)
        return cache["lf5"]

    def from_lf(series5: pd.Series) -> pd.Series:
        logc5, idx = lf5()
        vals = np.full(len(g), np.nan)
        ok = idx >= 0
        vals[ok] = series5.to_numpy(float)[idx[ok]]
        return pd.Series(vals, index=g.index)

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
        elif fam in ("P1", "P2", "P3"):
            d, s_, ag = psar(k)
            if fam == "P1":
                out[name] = pd.Series(d, index=g.index)
            elif fam == "P2":
                out[name] = _safe_log(c / pd.Series(s_, index=g.index).where(lambda x: x > 0))
            else:
                out[name] = np.log1p(pd.Series(ag, index=g.index))
        elif fam == "Q":
            rvk = _rv(r1, k)
            med = rvk.rolling(2880, min_periods=2880).median()
            out[name] = _safe_log(rvk / med.where(med > 0))
        elif fam in ("R", "S"):
            if lf5() is None:
                out[name] = pd.Series(np.nan, index=g.index)
            else:
                logc5, _ = lf5()
                out[name] = from_lf(logc5.diff(k) if fam == "R" else logc5 - _ema(logc5, k))
    df = pd.DataFrame(out, index=g.index).replace([np.inf, -np.inf], np.nan)
    return df[list(want)].astype(np.float32)


def short_pool_report() -> dict:
    """Bảng máy đọc được: họ → cửa sổ dùng / cửa sổ bỏ + lý do chính xác; tổng số cột; ngoại lệ DOW."""
    return {"grid": list(SHORT_GRID), "n_candidates": len(SHORT_CANDIDATES),
            "rule": "Candidate_m = C_short \\ overlap(C_short, S0_m) per model; tương quan cao chỉ báo cáo, không tự bỏ",
            "families": {fam: {"use": [list(u) if isinstance(u, tuple) else u for u in spec["use"]],
                               "skip": {str(k): v for k, v in spec["skip"].items()}} for fam, spec in SHORT_FAMILIES.items()},
            "columns": list(SHORT_COLUMNS)}
