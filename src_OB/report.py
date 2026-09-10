"""DATA_REPORT for the real prepared archive: raw continuity, replay segments, folds and origin masks.

Reads the downloaded Parquet, the prepared metadata/memmaps and the pipeline's own origin selection.
It does not rerun reconstruction, fit or infer any model, or create synthetic data.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json
from .data import Data, date_string, select_origins

QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)


def utc(us):
    return date_string(int(us))


def quantiles(values):
    values = np.asarray(values, np.float64)
    return {f"q{q:g}": round(float(np.quantile(values, q)), 3) for q in QUANTILES} if len(values) else {}


def raw_continuity(cfg, files):
    """Depth runs under the replay policy: next U <= previous u + 1 and 0 <= time step <= max_feed_gap."""
    import duckdb

    raw = Path(cfg["raw_dir"])
    paths = {kind: [str(raw / p) for p in files if p.startswith(kind + "/")] for kind in ("depth", "snapshots")}
    gap_ms = int(cfg["max_feed_gap_seconds"] * 1000)
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TEMP TABLE msg AS SELECT timestamp_ms ts, first_update_id f, last_update_id l, count(*) n "
            "FROM read_parquet(?) WHERE exchange = ? AND asset = ? GROUP BY 1, 2, 3",
            [paths["depth"], cfg["exchange"], cfg["symbol"]])
        runs = connection.execute(f"""
            WITH o AS (SELECT *, lag(l) OVER w AS pl, lag(ts) OVER w AS pts FROM msg WINDOW w AS (ORDER BY l, f, ts)),
            r AS (SELECT *, sum(CASE WHEN pl IS NULL OR f > pl + 1 OR ts - pts > {gap_ms} OR ts < pts THEN 1 ELSE 0 END)
                  OVER (ORDER BY l, f, ts ROWS UNBOUNDED PRECEDING) AS run FROM o)
            SELECT run, min(ts) AS t0, max(ts) AS t1, count(*) AS messages, sum(n) AS rows_
            FROM r GROUP BY run ORDER BY run""").df()
        anchors = []
        for ts, s in connection.execute(
                "SELECT timestamp_ms, last_update_id FROM read_parquet(?) WHERE exchange = ? AND asset = ? "
                "GROUP BY 1, 2 ORDER BY 1", [paths["snapshots"], cfg["exchange"], cfg["symbol"]]).fetchall():
            bridge = connection.execute("SELECT ts FROM msg WHERE f <= ? AND l >= ? ORDER BY l LIMIT 1",
                                        [s + 1, s + 1]).fetchone()
            following = connection.execute("SELECT ts, f FROM msg WHERE l > ? ORDER BY l LIMIT 1", [s]).fetchone()
            anchors.append({"snapshot_utc": utc(ts * 1000), "last_update_id": int(s),
                            "bridging_message_after_s": None if bridge is None else (bridge[0] - ts) / 1000,
                            "next_message_after_s": None if following is None else (following[0] - ts) / 1000,
                            "update_ids_missing": None if following is None else max(0, int(following[1] - s - 1))})
    finally:
        connection.close()
    t0, t1 = runs["t0"].to_numpy(np.int64), runs["t1"].to_numpy(np.int64)
    seconds = (t1 - t0) / 1000
    minutes = Counter(pd.to_datetime(t0, unit="ms", utc=True).minute)
    return {"depth_messages": int(runs["messages"].sum()), "depth_rows": int(runs["rows_"].sum()),
            "first_message_utc": utc(t0.min() * 1000), "last_message_utc": utc(t1.max() * 1000),
            "runs": len(runs), "run_seconds": quantiles(seconds),
            "gap_between_runs_seconds": quantiles((t0[1:] - t1[:-1]) / 1000),
            "covered_hours": round(float(seconds.sum()) / 3600, 3),
            "span_hours": round(float(t1.max() - t0.min()) / 3.6e6, 3),
            "run_start_minute_counts": {int(k): int(v) for k, v in sorted(minutes.items())},
            "snapshots_bridged": sum(a["bridging_message_after_s"] is not None for a in anchors),
            "snapshots": anchors}


def overlap_seconds(segments, start, end):
    return sum(max(0, min(s["end_exclusive_us"], end) - max(s["start_us"], start)) for s in segments) / 1e6


def data_report(cfg):
    folder = Path(cfg["prepared_dir"])
    manifest = json.loads((folder / "manifest.json").read_text())
    segments = json.loads((folder / "segments.json").read_text())
    reconstruction = json.loads((folder / "reconstruction.json").read_text())
    data = Data(cfg, require_current_replay=False)  # reporting may describe an older prepared version
    raw = raw_continuity(cfg, manifest["files"])
    kept = np.bincount(np.asarray(data.segment), minlength=len(segments))
    seg_rows = [{"id": s["id"], "start_utc": utc(s["start_us"]), "last_utc": utc(s["last_us"]),
                 "seconds": (s["last_us"] - s["start_us"]) / 1e6, "states": s["states"],
                 "kept_origins": int(kept[s["id"]]), "end_reason": s["end_reason"]} for s in segments]
    try:
        folds, fold_error = list(data.folds()), None
    except ValueError as exc:
        folds, fold_error = [], str(exc)
    fold_rows = []
    for fold in folds:
        counts = {}
        train_ids, val_ids = select_origins(cfg, data, fold, counts)
        fold_rows.append({"fold": fold.name, "train_start": utc(fold.train_start), "train_end": utc(fold.train_end),
                          "val_start": utc(fold.val_start), "val_end": utc(fold.val_end),
                          "valid_segment_seconds_fit": overlap_seconds(segments, fold.train_start, fold.train_end),
                          "valid_segment_seconds_val": overlap_seconds(segments, fold.val_start, fold.val_end),
                          **counts, "train_origins": len(train_ids), "val_origins": len(val_ids)})
    points = max(cfg["tfm"]["context"] if any(m.startswith("tfm") for m in cfg["models"]) else 1,
                 cfg["autots"]["max_window_size"] if "autots" in cfg["models"] else 1)
    need = [{"horizon_seconds": h, "price_context_points": points,
             "price_context_span_hours": round((points - 1) * h / 3600, 3),
             "segment_hours_needed_with_label": round(points * h / 3600, 3)} for h in cfg["horizons_seconds"]]
    blocked = [r["fold"] for r in fold_rows if not r["train_origins"] or not r["val_origins"]]
    status = "BLOCKED" if fold_error or blocked else "READY"
    report = {"status": status, "blocked_folds": blocked, "fold_error": fold_error,
              "dataset": {"repo": manifest["dataset_repo"], "revision": manifest["dataset_revision"],
                          "files": manifest["files"], "prepared_dir": str(folder),
                          "schema_version": manifest["schema_version"]},
              "raw_archive": raw,
              "prepared": {"coverage": manifest["coverage"], "counts": manifest["counts"],
                           "reconstruction_counts": reconstruction["counts"],
                           "reset_reasons": dict(Counter(r["reason"] for r in reconstruction["resets"])),
                           "known_hard_gaps_utc": [[utc(a), utc(b)] for a, b in reconstruction["known_hard_gaps_us"]],
                           "segments": len(segments), "segment_seconds": quantiles([r["seconds"] for r in seg_rows]),
                           "segment_states": quantiles([r["states"] for r in seg_rows]),
                           "segment_end_reasons": dict(Counter(r["end_reason"] for r in seg_rows)),
                           "segment_table": seg_rows},
              "folds": fold_rows, "context_requirements": need}
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_report.json", report)
    (output / "DATA_REPORT.md").write_text(markdown(cfg, report))
    print(f"DATA_REPORT status={status}; folds={len(fold_rows)}; blocked={blocked or fold_error}", flush=True)


def table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in columns) + " |" for r in rows]
    return lines


def markdown(cfg, report):
    raw, prep = report["raw_archive"], report["prepared"]
    coverage = prep["coverage"]
    longest = max((r["seconds"] for r in prep["segment_table"]), default=0.)
    lines = [
        "# DATA_REPORT — BTCUSDT Binance Spot L2 từ HF archive pinned", "",
        "Sinh bởi `python -m src_OB data-report` từ raw Parquet đã tải và prepared dataset thật "
        "(DuckDB đọc raw; memmap/metadata của `prepare`; mask origin dùng chung hàm `select_origins` với `train`). "
        "Không fit/infer model, không dữ liệu tổng hợp.", "",
        f"**Trạng thái: {report['status']}**" + (f" — fold không có origin FIT/VAL hợp lệ: {', '.join(report['blocked_folds'])}"
                                                   if report["blocked_folds"] else "")
        + (f" — {report['fold_error']}" if report["fold_error"] else ""), "",
        "## 1. Nguồn", "",
        f"- `{report['dataset']['repo']}` revision `{report['dataset']['revision']}`; prepared schema "
        f"v{report['dataset']['schema_version']} tại `{Path(report['dataset']['prepared_dir']).relative_to(Path(cfg['raw_dir']).parents[2])}`.",
        f"- File: {', '.join('`' + f + '`' for f in report['dataset']['files'])}.", "",
        "## 2. Depth diff thô trước replay", "",
        f"- {raw['depth_messages']:,} message ({raw['depth_rows']:,} row), từ {raw['first_message_utc']} tới {raw['last_message_utc']}.",
        f"- Chia theo đúng policy replay (U ≤ u trước + 1, bước thời gian 0–{cfg['max_feed_gap_seconds']} s): "
        f"**{raw['runs']:,} run liên tục**. Tổng thời gian có depth **{raw['covered_hours']} h** trên span {raw['span_hours']} h "
        f"({100 * raw['covered_hours'] / raw['span_hours']:.1f}%).",
        f"- Độ dài run (giây): {raw['run_seconds']}.",
        f"- Khoảng trống giữa hai run (giây): {raw['gap_between_runs_seconds']}.",
        f"- Phút UTC bắt đầu run: {raw['run_start_minute_counts']}.",
        f"- Snapshot: {len(raw['snapshots'])}; snapshot có depth message nối được `last_update_id + 1`: **{raw['snapshots_bridged']}**.", "",
        *table(raw["snapshots"], ["snapshot_utc", "last_update_id", "bridging_message_after_s",
                                  "next_message_after_s", "update_ids_missing"]), "",
        "## 3. Replay thật (`prepare`)", "",
        f"- Raw state: **{prep['counts']['raw']:,}**; origin giữ lại sau same-mid drop: **{prep['counts']['kept']:,}** "
        "(mỗi segment tính cả state snapshot đầu tiên).",
        f"- Coverage theo segment: {utc(coverage['start_inclusive_us'])} → {utc(coverage['end_exclusive_us'] - 1)}; "
        f"tổng thời gian segment hợp lệ **{coverage['valid_segment_microseconds'] / 1e6:.3f} s**; segment dài nhất {longest:.3f} s.",
        f"- Đếm replay: {prep['reconstruction_counts']}.",
        f"- Lý do kết thúc segment: {prep['segment_end_reasons']}; reset: {prep['reset_reasons']}.",
        f"- Hard gap đã biết: {prep['known_hard_gaps_utc']} (không replay xuyên qua; event trong khoảng này bị bỏ).",
        f"- Segment: {prep['segments']}; thời lượng (s) {prep['segment_seconds']}; state/segment {prep['segment_states']}.", "",
        *table(prep["segment_table"], ["id", "start_utc", "last_utc", "seconds", "states", "kept_origins", "end_reason"]), "",
        "## 4. Walk-forward và origin theo mask của pipeline", "",
        f"Fold lấy từ coverage theo lịch (train {cfg['train_days']} d, gap {cfg['gap_days']} d, VAL {cfg['val_days']} d, "
        f"bước {cfg['step_days']} d, tối đa {cfg['n_folds']}). Số đếm là origin còn lại sau từng mask, "
        "cộng dồn qua các horizon (tập chung cho mọi family).", "",
    ]
    if report["folds"]:
        columns = ["fold", "train_start", "train_end", "val_start", "val_end", "valid_segment_seconds_fit",
                   "valid_segment_seconds_val", "train_label_context", "train_context_inside_fit", "val_label_context"]
        columns += [k for k in report["folds"][0] if k.startswith(("train_price", "val_price"))]
        lines += table(report["folds"], columns + ["train_origins", "val_origins"])
    else:
        lines.append(f"Không tạo được fold: {report['fold_error']}")
    lines += ["", "## 5. Context cần so với segment thực tế", "",
              *table(report["context_requirements"], ["horizon_seconds", "price_context_points",
                                                      "price_context_span_hours", "segment_hours_needed_with_label"]), "",
              f"LSTM/tree cần {cfg['context']} origin mid-change liên tiếp cùng segment và nhãn tại t+h trong segment; "
              f"TimesFM/AutoTS cần thêm context giá cách đều h như bảng. Segment hợp lệ dài nhất: {longest:.3f} s; "
              f"run depth thô dài nhất: {raw['run_seconds'].get('q1')} s.", ""]
    return "\n".join(lines) + "\n"
