"""Data: đọc CSV, adapter lowercase → B0 uppercase (không sửa B0), kiểm tra §1.1, checksum, as-of join LF 5'."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STEP_SEC

RAW_COLUMNS = ("datetime", "timestamp", "open", "high", "low", "close", "volume", "amount")
B0_COLUMNS = ("timestamp", "Open", "High", "Low", "Close", "Volume")


def read_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Đọc CSV raw (lowercase). Dòng cuối cụt (file cắt 2 MiB) bị bỏ qua nhờ on_bad_lines + dropna.

    Alias (data 2 năm 2026-09-04): cột `ts` được đổi tên thành `timestamp` TRONG BỘ NHỚ khi file không có `timestamp`;
    có cả hai thì phải giống hệt nhau (khác → lỗi). Cột `datetime` nếu có phải khớp epoch của `timestamp` (UTC) — sau đó
    `datetime` luôn được dựng lại từ `timestamp` để mọi đường downstream chỉ dựa vào epoch."""
    df = pd.read_csv(path, on_bad_lines="skip")
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    elif "timestamp" in df.columns and "ts" in df.columns:
        a = pd.to_numeric(df["ts"], errors="coerce")
        b = pd.to_numeric(df["timestamp"], errors="coerce")
        if not a.fillna(-1).eq(b.fillna(-1)).all():
            raise ValueError(f"{path}: cột `ts` và `timestamp` không giống nhau — không đoán, dừng")
        df = df.drop(columns=["ts"])
    missing = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing} trong {path}")
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(size - 1, 0))
        ends_with_newline = f.read(1) == b"\n"
    if not ends_with_newline and len(df):
        df = df.iloc[:-1]  # dòng cuối cụt (file bị cắt 2 MiB) có thể parse thành số sai → bỏ hẳn
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    else:
        df["amount"] = np.nan
    n_before = len(df)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if "datetime" in df.columns:  # verify cột datetime của file khớp epoch (UTC) — không tin cột chữ, chỉ kiểm
        parsed = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        epoch = (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()  # độc lập resolution (pandas 3: s/us/ns)
        bad = parsed.isna() | (epoch != df["timestamp"].to_numpy(np.int64))
        if bad.any():
            i = int(np.flatnonzero(bad.to_numpy())[0])
            raise ValueError(f"{path}: cột datetime không khớp timestamp (UTC) tại dòng {i}: {df['datetime'].iloc[i]!r} vs {int(df['timestamp'].iloc[i])}")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.attrs["duplicates_dropped"] = int(n_before - len(df))  # drop xảy ra trước check → check_ohlcv cộng vào `duplicates`
    return df


# ----------------------------------------------------------------------------- LF 5 phút dẫn xuất từ HF (2026-09-04)
LF_COLUMNS = ("datetime", "timestamp", "open", "high", "low", "close", "volume", "amount")


def derive_lf_5min(hf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bar 5 phút ĐÃ ĐÓNG từ bar 1 phút, đúng quy ước LF của project: nhãn T = timestamp bar 1' cuối của nhóm (T−4' … T],
    T là bội của 300 s; open = first, high = max, low = min, close = last, volume = sum, amount = sum.

    Chỉ giữ nhóm ĐỦ 5 bar liên tiếp (60 s); nhóm đầu/cuối thiếu bar bị BỎ (không bịa bar). Bar T chỉ chứa thông tin ≤ T,
    và chỉ được as-of join vào origin t khi T ≤ t (`asof_index`) → không look-ahead. Trả (lf, meta)."""
    ts = hf["timestamp"].to_numpy(np.int64)
    if len(ts) == 0 or ((ts % STEP_SEC) != 0).any():
        raise ValueError("HF phải nằm trên lưới 60 s để dẫn xuất LF 5'")
    grp = ((ts // 60) - 1) // 5  # bar (T−4 … T] cùng nhóm, T = bội 300 s (cùng quy ước `synthetic.make_lf` và data LF cũ)
    g = hf[["timestamp", "open", "high", "low", "close", "volume", "amount"]].copy()
    g["grp"] = grp
    size = g.groupby("grp")["timestamp"].transform("size")
    span = g.groupby("grp")["timestamp"].transform("max") - g.groupby("grp")["timestamp"].transform("min")
    full = g[(size == 5) & (span == 4 * STEP_SEC)]
    n_dropped = int(g["grp"].nunique() - full["grp"].nunique())
    agg = full.groupby("grp", sort=True).agg(timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"),
                                             low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), amount=("amount", "sum"))
    agg = agg.reset_index(drop=True)
    if len(agg) and ((agg["timestamp"] % 300) != 0).any():
        raise AssertionError("nhãn bar 5' phải là bội của 300 s")
    agg["datetime"] = pd.to_datetime(agg["timestamp"], unit="s", utc=True)
    lf = agg[list(LF_COLUMNS)]
    meta = {"method": "closed 5-minute buckets (T-4..T], label T = last 1m bar ts (multiple of 300 s); open=first, high=max, low=min, "
                      "close=last, volume=sum, amount=sum; incomplete first/last buckets dropped",
            "rows_hf": int(len(hf)), "rows_lf": int(len(lf)), "dropped_incomplete_buckets": n_dropped,
            "lf_start": str(lf["datetime"].iloc[0]) if len(lf) else None, "lf_end": str(lf["datetime"].iloc[-1]) if len(lf) else None}
    return lf, meta


def write_lf_csv(lf: pd.DataFrame, path: str | Path) -> str:
    """Ghi LF tất định (cùng input → cùng byte): UTF-8, '\\n', không index, datetime ISO UTC. Trả sha256 của file."""
    out = lf.copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        out.to_csv(f, index=False, lineterminator="\n")
    return file_sha256(path)


def check_ohlcv(df: pd.DataFrame, step: int = STEP_SEC) -> dict:
    """Kiểm tra §1.1: lưới, dup, gap, OHLC sanity, amount/volume trong [L, H]. Trả report (không raise)."""
    ts = df["timestamp"].to_numpy(np.int64)
    d = np.diff(ts)
    aligned = bool(((ts % step) == 0).all())
    dups = int((d == 0).sum()) + int(df.attrs.get("duplicates_dropped", 0))
    gaps = int((d > step).sum())
    ohlc_ok = bool(((df["high"] >= df[["open", "close"]].max(axis=1)) & (df["low"] <= df[["open", "close"]].min(axis=1))).all())
    vol_ok = bool((df["volume"] >= 0).all())
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = df["amount"].to_numpy(float) / df["volume"].to_numpy(float)
    m = np.isfinite(avg) & (df["volume"].to_numpy(float) > 0)
    amt_ok = bool(((avg[m] >= df["low"].to_numpy(float)[m] * (1 - 1e-6)) & (avg[m] <= df["high"].to_numpy(float)[m] * (1 + 1e-6))).all()) if m.any() else True
    rep = {
        "rows": int(len(df)),
        "start": str(df["datetime"].iloc[0]),
        "end": str(df["datetime"].iloc[-1]),
        "aligned": aligned,
        "duplicates": dups,
        "gaps": gaps,
        "max_gap_sec": int(d.max()) if len(d) else 0,
        "ohlc_ok": ohlc_ok,
        "volume_ok": vol_ok,
        "amount_in_range": amt_ok,
        "amount_available": bool(m.any()),
    }
    rep["ok"] = aligned and dups == 0 and gaps == 0 and ohlc_ok and vol_ok and amt_ok
    return rep


def to_b0_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adapter: lowercase → cột B0 (`timestamp, Open, High, Low, Close, Volume`). B0 tự reindex lưới đầy đủ."""
    out = pd.DataFrame({
        "timestamp": df["timestamp"].to_numpy(np.int64),
        "Open": df["open"].to_numpy(float),
        "High": df["high"].to_numpy(float),
        "Low": df["low"].to_numpy(float),
        "Close": df["close"].to_numpy(float),
        "Volume": df["volume"].to_numpy(float),
    })
    return out


def grid_frame(b0_frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Lưới 1 phút đầy đủ (frame của B0 sau prepare_minute_ohlcv) + amount, cột lowercase, index UTC — dùng để tính ext feature."""
    ts = b0_frame["timestamp"].to_numpy(np.int64)
    g = pd.DataFrame({"timestamp": ts})
    for c in ("Open", "High", "Low", "Close", "Volume"):
        g[c.lower()] = b0_frame[c].to_numpy(float)
    amt = pd.Series(raw["amount"].to_numpy(float), index=raw["timestamp"].to_numpy(np.int64))
    g["amount"] = amt.reindex(ts).to_numpy(float)
    g.index = pd.to_datetime(ts, unit="s", utc=True)
    return g


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(label: str, files: dict[str, Path], reports: dict[str, dict], out_path: Path, root: Path | None = None) -> dict:
    """§6.1: sha256 + report từng file; path ghi TƯƠNG ĐỐI so với root (posix) để verify được trên máy khác (Vast)."""
    payload = {"dataset_label": label, "files": {}}
    for name, p in files.items():
        p = Path(p)
        rel = p.resolve()
        if root is not None:
            try:
                rel = rel.relative_to(Path(root).resolve())
            except ValueError:
                pass
        payload["files"][name] = {"path": rel.as_posix(), "sha256": file_sha256(p), "bytes": p.stat().st_size, "report": reports.get(name, {})}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def verify_checksums(checksum_path: Path, root: Path, label: str | None = None) -> tuple[bool, list[str]]:
    """So sha256 của file hiện có với data_checksums.json (và dataset_label nếu truyền). Trả (ok, danh sách vấn đề)."""
    payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    problems = []
    if label is not None and payload.get("dataset_label") != label:
        problems.append(f"dataset_label khác: checksum '{payload.get('dataset_label')}' vs config '{label}'")
    for name, info in payload["files"].items():
        p = Path(info["path"])
        p = p if p.is_absolute() else Path(root) / p
        if not p.exists():
            problems.append(f"{name}: thiếu file {p}")
            continue
        if file_sha256(p) != info["sha256"]:
            problems.append(f"{name}: sha256 khác {p}")
    return (not problems), problems


def asof_index(lf_ts: np.ndarray, hf_ts: np.ndarray) -> np.ndarray:
    """Với mỗi t của lưới HF: chỉ số bar LF có nhãn T ≤ t (bar 5' đã đóng tại origin t); −1 nếu chưa có."""
    return np.searchsorted(lf_ts, hf_ts, side="right") - 1
