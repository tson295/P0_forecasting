"""39 candidate feature §2.3 — mỗi cột một hàm, causal (chỉ dùng bar ≤ t, cửa sổ kết thúc tại t), lookback ≤ 1440'
(ngoại lệ #35 log_rv60_med2d dùng median 2880' đúng như định nghĩa trong plan).

Input: lưới 1 phút đầy đủ `g` (index UTC; cột lowercase open/high/low/close/volume/amount; NaN tại gap) và LF 5' (tuỳ chọn).
Output: DataFrame các cột ext trên cùng lưới (NaN ở warmup). Tree nhận NaN native; LSTM điền 0 sau chuẩn hoá (harness).
Ký hiệu: A = amount (quote volume) → A/V là VWAP thật của bar; TP = (H+L+C)/3; r1 = Δlog C; rv_k = sqrt(mean_k r1²).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Baseline_LGBM import _ema, _hma, _rsi  # helper causal của B0 (import như library)

from .data import asof_index


@dataclass(frozen=True)
class Candidate:
    name: str
    columns: tuple[str, ...]
    group: str


# Thứ tự thử = thứ tự §2.3
CANDIDATES: list[Candidate] = [
    Candidate("vwap_amt_gap_1", ("vwap_amt_gap_1",), "A"),
    Candidate("vwap_amt_gap_15", ("vwap_amt_gap_15",), "A"),
    Candidate("vwap_amt_gap_60", ("vwap_amt_gap_60",), "A"),
    Candidate("vwap_amt_gap_240", ("vwap_amt_gap_240",), "A"),
    Candidate("ret_60", ("ret_60",), "B"),
    Candidate("ret_240", ("ret_240",), "B"),
    Candidate("ret_1440", ("ret_1440",), "B"),
    Candidate("log_rv15_rv240", ("log_rv15_rv240",), "B"),
    Candidate("log_rv60_rv1440", ("log_rv60_rv1440",), "B"),
    Candidate("ret_skew_60", ("ret_skew_60",), "B"),
    Candidate("dd_240", ("dd_240",), "B"),
    Candidate("ru_240", ("ru_240",), "B"),
    Candidate("log_c_ema60", ("log_c_ema60",), "C"),
    Candidate("log_c_ema240", ("log_c_ema240",), "C"),
    Candidate("log_c_ema1440", ("log_c_ema1440",), "C"),
    Candidate("log_ema60_ema240", ("log_ema60_ema240",), "C"),
    Candidate("hma_slope64_volnorm", ("hma_slope64_volnorm",), "C"),
    Candidate("rsi240_centered", ("rsi240_centered",), "D"),
    Candidate("macd_hist_60_240_60_volnorm", ("macd_hist_60_240_60_volnorm",), "D"),
    Candidate("bb_pctb_20", ("bb_pctb_20",), "E"),
    Candidate("bb_pctb_60", ("bb_pctb_60",), "E"),
    Candidate("bb_logbw_20", ("bb_logbw_20",), "E"),
    Candidate("log_atr14_c", ("log_atr14_c",), "F"),
    Candidate("log_atr14_rv14", ("log_atr14_rv14",), "F"),
    Candidate("kcw_20", ("kcw_20",), "F"),
    Candidate("mfi14_centered", ("mfi14_centered",), "G"),
    Candidate("mfi60_centered", ("mfi60_centered",), "G"),
    Candidate("ad_vwclv_5", ("ad_vwclv_5",), "H"),
    Candidate("ad_vwclv_15", ("ad_vwclv_15",), "H"),
    Candidate("ad_vwclv_60", ("ad_vwclv_60",), "H"),
    Candidate("psar_dir", ("psar_dir",), "I"),
    Candidate("psar_logdist", ("psar_logdist",), "I"),
    Candidate("psar_age_log", ("psar_age_log",), "I"),
    Candidate("dow", ("dow_sin", "dow_cos"), "J"),
    Candidate("log_rv60_med2d", ("log_rv60_med2d",), "J"),
    Candidate("log_range_240", ("log_range_240",), "J"),
    Candidate("r5_1", ("r5_1",), "K"),
    Candidate("r5_12", ("r5_12",), "K"),
    Candidate("log_c5_ema5_12", ("log_c5_ema5_12",), "K"),
]
CANDIDATE_BY_NAME = {c.name: c for c in CANDIDATES}
ALL_EXT_COLUMNS = tuple(col for c in CANDIDATES for col in c.columns)


def _safe_log(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    return np.log(x.where(x > 0))


def _rsum(x: pd.Series, k: int) -> pd.Series:
    return x.rolling(k, min_periods=k).sum()


def _rv(r1: pd.Series, k: int) -> pd.Series:
    return np.sqrt((r1 ** 2).rolling(k, min_periods=k).mean())


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    tr = tr.where(prev.notna())
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def psar(high: np.ndarray, low: np.ndarray, close: np.ndarray, af0: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
    """Parabolic SAR (Wilder). Trả (dir, sar_next, age): sar_next[t] = mức SAR tính từ bar ≤ t; reset tại gap NaN."""
    n = len(close)
    direction = np.full(n, np.nan)
    sar_next = np.full(n, np.nan)
    age = np.full(n, np.nan)
    valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        _psar_segment(high[i:j], low[i:j], close[i:j], direction[i:j], sar_next[i:j], age[i:j], af0, af_step, af_max)
        i = j
    return direction, sar_next, age


def _psar_segment(h, l, c, dir_out, sar_out, age_out, af0, af_step, af_max):
    n = len(c)
    if n < 2:
        return
    up = bool(c[1] >= c[0])
    sar = l[0] if up else h[0]
    ep = h[0] if up else l[0]
    af = af0
    since = 0
    for i in range(1, n):
        if up:
            if l[i] < sar:  # flip → down
                up, sar, ep, af, since = False, ep, l[i], af0, 0
            else:
                if h[i] > ep:
                    ep, af = h[i], min(af + af_step, af_max)
                since += 1
        else:
            if h[i] > sar:  # flip → up
                up, sar, ep, af, since = True, ep, h[i], af0, 0
            else:
                if l[i] < ep:
                    ep, af = l[i], min(af + af_step, af_max)
                since += 1
        nxt = sar + af * (ep - sar)
        nxt = min(nxt, l[i], l[i - 1]) if up else max(nxt, h[i], h[i - 1])
        sar = nxt
        dir_out[i] = 1.0 if up else -1.0
        sar_out[i] = sar
        age_out[i] = since


def compute_ext(g: pd.DataFrame, lf: pd.DataFrame | None = None, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Tính các cột ext (mặc định tất cả) trên lưới g. Mọi cửa sổ kết thúc tại t; không dùng bar > t."""
    want = set(columns) if columns is not None else set(ALL_EXT_COLUMNS)
    out: dict[str, pd.Series] = {}
    c, o, h, l, v, a = (g[k].astype(float) for k in ("close", "open", "high", "low", "volume", "amount"))
    logc = _safe_log(c)
    r1 = logc.diff()
    rv60 = _rv(r1, 60)
    rv60p = rv60.where(rv60 > 0)

    def need(*names: str) -> bool:
        return any(nm in want for nm in names)

    # A. VWAP thật từ amount
    for k in (1, 15, 60, 240):
        nm = f"vwap_amt_gap_{k}"
        if nm in want:
            vw = (a / v.where(v > 0)) if k == 1 else (_rsum(a, k) / _rsum(v, k).where(_rsum(v, k) > 0))
            out[nm] = _safe_log(c / vw.where(vw > 0))
    # B. return / rolling stats
    for k in (60, 240, 1440):
        nm = f"ret_{k}"
        if nm in want:
            out[nm] = logc.diff(k)
    if "log_rv15_rv240" in want:
        out["log_rv15_rv240"] = _safe_log(_rv(r1, 15) / _rv(r1, 240).where(_rv(r1, 240) > 0))
    if "log_rv60_rv1440" in want:
        rv1440 = _rv(r1, 1440)
        out["log_rv60_rv1440"] = _safe_log(rv60 / rv1440.where(rv1440 > 0))
    if "ret_skew_60" in want:
        out["ret_skew_60"] = r1.rolling(60, min_periods=60).skew()
    if "dd_240" in want:
        out["dd_240"] = _safe_log(c / c.rolling(240, min_periods=240).max())
    if "ru_240" in want:
        out["ru_240"] = _safe_log(c / c.rolling(240, min_periods=240).min())
    # C. EMA / HMA
    for k in (60, 240, 1440):
        nm = f"log_c_ema{k}"
        if nm in want:
            out[nm] = logc - _ema(logc, k)
    if "log_ema60_ema240" in want:
        out["log_ema60_ema240"] = _ema(logc, 60) - _ema(logc, 240)
    if "hma_slope64_volnorm" in want:
        out["hma_slope64_volnorm"] = _hma(logc, 64).diff() / rv60p
    # D. RSI / MACD
    if "rsi240_centered" in want:
        out["rsi240_centered"] = _rsi(r1, 240) / 100.0 - 0.5
    if "macd_hist_60_240_60_volnorm" in want:
        macd = _ema(logc, 60) - _ema(logc, 240)
        out["macd_hist_60_240_60_volnorm"] = (macd - _ema(macd, 60)) / rv60p
    # E. Bollinger trên log C
    for n in (20, 60):
        nm = f"bb_pctb_{n}"
        if nm in want or (n == 20 and "bb_logbw_20" in want):
            sma = logc.rolling(n, min_periods=n).mean()
            sd = logc.rolling(n, min_periods=n).std(ddof=0)
            if nm in want:
                out[nm] = (logc - sma) / (2.0 * sd.where(sd > 0))
            if n == 20 and "bb_logbw_20" in want:
                out["bb_logbw_20"] = _safe_log(sd)
    # F. ATR / Keltner
    if need("log_atr14_c", "log_atr14_rv14"):
        atr14 = _atr(h, l, c, 14)
        if "log_atr14_c" in want:
            out["log_atr14_c"] = _safe_log(atr14 / c)
        if "log_atr14_rv14" in want:
            rv14 = _rv(r1, 14)
            out["log_atr14_rv14"] = _safe_log((atr14 / c) / rv14.where(rv14 > 0))
    if "kcw_20" in want:
        ema20c = c.ewm(span=20, adjust=False, min_periods=20).mean()
        out["kcw_20"] = _safe_log(2.0 * _atr(h, l, c, 20) / ema20c.where(ema20c > 0))
    # G. MFI với money flow = amount
    if need("mfi14_centered", "mfi60_centered"):
        tp = (h + l + c) / 3.0
        dtp = tp.diff()
        a_pos = a.where(dtp > 0, 0.0).where(dtp.notna())
        a_neg = a.where(dtp < 0, 0.0).where(dtp.notna())
        for n in (14, 60):
            nm = f"mfi{n}_centered"
            if nm in want:
                pos, neg = _rsum(a_pos, n), _rsum(a_neg, n)
                den = pos + neg
                out[nm] = pos / den.where(den > 0) - 0.5
    # H. A/D rolling (không tích luỹ)
    if need("ad_vwclv_5", "ad_vwclv_15", "ad_vwclv_60"):
        rng_ = h - l
        clv = (((c - l) - (h - c)) / rng_.where(rng_ > 0)).fillna(0.0).where(rng_.notna())
        for k in (5, 15, 60):
            nm = f"ad_vwclv_{k}"
            if nm in want:
                out[nm] = _rsum(clv * v, k) / _rsum(v, k).where(_rsum(v, k) > 0)
    # I. Parabolic SAR
    if need("psar_dir", "psar_logdist", "psar_age_log"):
        d, s, ag = psar(h.to_numpy(float), l.to_numpy(float), c.to_numpy(float))
        if "psar_dir" in want:
            out["psar_dir"] = pd.Series(d, index=g.index)
        if "psar_logdist" in want:
            out["psar_logdist"] = _safe_log(c / pd.Series(s, index=g.index).where(lambda x: x > 0))
        if "psar_age_log" in want:
            out["psar_age_log"] = np.log1p(pd.Series(ag, index=g.index))
    # J. regime / calendar
    if need("dow_sin", "dow_cos"):
        wd = pd.Series(g.index.dayofweek.to_numpy(float), index=g.index)
        if "dow_sin" in want:
            out["dow_sin"] = np.sin(2 * np.pi * wd / 7.0)
        if "dow_cos" in want:
            out["dow_cos"] = np.cos(2 * np.pi * wd / 7.0)
    if "log_rv60_med2d" in want:
        med = rv60.rolling(2880, min_periods=2880).median()
        out["log_rv60_med2d"] = _safe_log(rv60 / med.where(med > 0))
    if "log_range_240" in want:
        out["log_range_240"] = _safe_log((h.rolling(240, min_periods=240).max() - l.rolling(240, min_periods=240).min()) / c)
    # K. resolution 5 phút (as-of join, chỉ bar đã đóng T ≤ t)
    if need("r5_1", "r5_12", "log_c5_ema5_12"):
        if lf is None:
            for nm in ("r5_1", "r5_12", "log_c5_ema5_12"):
                if nm in want:
                    out[nm] = pd.Series(np.nan, index=g.index)
        else:
            lf = lf.sort_values("timestamp").reset_index(drop=True)
            c5 = lf["close"].astype(float)
            logc5 = _safe_log(c5)
            feats = {"r5_1": logc5.diff(), "r5_12": logc5.diff(12), "log_c5_ema5_12": logc5 - _ema(logc5, 12)}
            idx = asof_index(lf["timestamp"].to_numpy(np.int64), g["timestamp"].to_numpy(np.int64))
            ok = idx >= 0
            for nm, series in feats.items():
                if nm in want:
                    vals = np.full(len(g), np.nan)
                    vals[ok] = series.to_numpy(float)[idx[ok]]
                    out[nm] = pd.Series(vals, index=g.index)
    df = pd.DataFrame(out, index=g.index)
    return df[[col for col in (columns or ALL_EXT_COLUMNS) if col in df.columns]].astype(np.float32)
