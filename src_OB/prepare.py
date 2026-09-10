"""Reconstruct historical books, then create OF/OFI and separate raw-mid memmaps."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .config import write_json
from .reconstruct import BookReplay, REPLAY_VERSION, states

DTYPES = {"raw_ts": "int64", "raw_mid": "float64", "raw_segment": "int64",
          "ts": "int64", "mid": "float64", "segment": "int64", "features": "float32"}


def feature_names():
    return ([f"{field}_{level}" for level in range(10) for field in ("of_bid", "of_ask", "ofi")]
            + ["log_elapsed", "log_raw_elapsed", "log_updates"])


def prepare(cfg):
    raw = Path(cfg["raw_dir"])
    source = json.loads((raw / "download_manifest.json").read_text())
    if source.get("status") != "complete" or source.get("provider") != "huggingface":
        raise ValueError("Cần HF historical archive đã tải hoàn tất.")
    if source["repo"] != cfg["dataset_repo"] or source["revision"] != cfg["dataset_revision"]:
        raise ValueError("Archive revision không khớp cấu hình; không trộn data.")
    files = [f["path"] for f in source["selected_files"]]
    for name in files:
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    dest = Path(cfg["prepared_dir"])
    dest.mkdir(parents=True, exist_ok=False)
    replay = BookReplay(cfg)
    pending = np.zeros((10, 2), np.float64)
    previous = None
    last_kept_ts = None
    pending_count = 0
    totals = {"raw": 0, "kept": 0}
    raw_rows, kept_rows, features = [], [], []
    with ExitStack() as stack:
        handles = {key: stack.enter_context((dest / f"{key}.bin").open("wb")) for key in DTYPES}

        def flush():
            for rows, prefix in ((raw_rows, "raw_"), (kept_rows, "")):
                if rows:
                    for j, key in enumerate((prefix + "ts", prefix + "mid", prefix + "segment")):
                        np.asarray([row[j] for row in rows], dtype=DTYPES[key]).tofile(handles[key])
            if features:
                np.asarray(features, np.float32).tofile(handles["features"])
            totals["raw"] += len(raw_rows)
            totals["kept"] += len(kept_rows)
            raw_rows.clear()
            kept_rows.clear()
            features.clear()

        for now, segment, (bp, bq, ap, aq) in states(cfg, raw, dest / "replay_sort", replay, files):
            mid = (bp[0] + ap[0]) / 2  # exactly one mid-price per atomic book state
            if previous is None or previous[1] != segment:
                pending.fill(0)
                pending_count = 0
                last_kept_ts = now
                raw_dt = 0.
                keep = True
            else:
                old_ts, _, old_bp, old_bq, old_ap, old_aq, old_mid = previous
                bf = np.where(bp > old_bp, bq, np.where(bp == old_bp, bq - old_bq, -old_bq))
                af = np.where(ap < old_ap, aq, np.where(ap == old_ap, aq - old_aq, -old_aq))
                pending += np.stack((bf, af), axis=1)
                pending_count += 1
                raw_dt = (now - old_ts) / 1e6
                keep = mid != old_mid  # flow above is accumulated even when this is false
            raw_rows.append((now, mid, segment))
            if keep:
                level_features = np.column_stack((pending, pending[:, 0] - pending[:, 1])).reshape(-1)
                timing = np.log1p([(now - last_kept_ts) / 1e6, raw_dt, pending_count])
                features.append(np.concatenate((level_features, timing)))
                kept_rows.append((now, mid, segment))
                last_kept_ts = now
                pending.fill(0)
                pending_count = 0
            previous = (now, segment, bp, bq, ap, aq, mid)
            if len(raw_rows) >= cfg["chunk_rows"]:
                flush()
                print(f"reconstructed states={totals['raw']:,}, origins={totals['kept']:,}", flush=True)
        flush()
    if not totals["raw"]:
        raise ValueError("Không có state top-10 reconstructable; không giả lập lịch sử.")
    coverage = {"start_inclusive_us": replay.segments[0]["start_us"],
                "end_exclusive_us": replay.segments[-1]["end_exclusive_us"],
                "valid_segment_microseconds": sum(s["end_exclusive_us"] - s["start_us"] for s in replay.segments),
                "source_first_observed_us": replay.source_first, "source_last_observed_us": replay.source_last}
    write_json(dest / "segments.json", replay.segments)
    write_json(dest / "reconstruction.json", {"counts": dict(replay.counts), "resets": replay.resets,
               "known_hard_gaps_us": replay.known_gaps, "replay_version": REPLAY_VERSION,
               "collector_warning": "June-August 2026 may contain missing updates; IDs/timestamps cannot certify undetectable omissions."})
    write_json(dest / "manifest.json", {"schema_version": 3, "replay_version": REPLAY_VERSION, "config": cfg,
               "historical_fixed": True,
               "dataset_repo": source["repo"], "dataset_revision": source["revision"], "coverage": coverage,
               "counts": totals, "features": feature_names(), "dtypes": DTYPES, "files": files,
               "timestamp_unit": "microseconds (source timestamp_ms * 1000)",
               "timestamp_semantics": "dataset event/receipt timestamp; legacy collector does not guarantee arrival time",
               "price": "L2 best-bid/best-ask mid", "of": "price-aware, accumulated before same-mid filtering",
               "source": "HF snapshot + sequenced absolute-quantity depth replay", "sequence_ids_available": True})
    print(f"Prepared observed coverage: {coverage}", flush=True)
