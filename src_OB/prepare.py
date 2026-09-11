"""Reconstruct historical books (diff replay or full snapshots), then create OF/OFI and raw-mid memmaps."""
from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .config import ROOT, write_json
from .reconstruct import BookReplay, REPLAY_VERSION, states

DTYPES = {"raw_ts": "int64", "raw_mid": "float64", "raw_segment": "int64",
          "ts": "int64", "mid": "float64", "segment": "int64", "features": "float32"}


def code_provenance(cfg):
    """Code commit, uncommitted pipeline/config paths and resolved-config hash behind a run."""
    def git(*args):
        try:
            # --no-optional-locks: never take index.lock next to a concurrent artifact commit.
            result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True,
                                    text=True, errors="replace", timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None

    head = git("rev-parse", "--verify", "HEAD")
    # NUL-separated porcelain v1 keeps paths unquoted; renames/copies carry the source path as an extra field.
    status = git("status", "--porcelain", "-z", "-uall", "--", "src_OB", "configs", "src/p0")
    paths = None
    if status is not None:
        paths, fields = [], iter(status.split("\0"))
        for entry in fields:
            if len(entry) > 3:
                paths.append(entry[3:])
                if {"R", "C"} & set(entry[:2]):  # rename/copy in index or work tree carries a source field
                    next(fields, None)
    return {"code_commit": head.strip() if head else None,
            "code_uncommitted_paths": paths,  # None means git was unavailable or failed
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}


def feature_names():
    return ([f"{field}_{level}" for level in range(10) for field in ("of_bid", "of_ask", "ofi")]
            + ["log_elapsed", "log_raw_elapsed", "log_updates"])


def source_stream(cfg, raw, dest, files):
    """Book-state stream, segment tracker and source notes for the configured provider."""
    if cfg["provider"] == "zenodo":
        from .snapshots import SNAPSHOT_ADAPTER_VERSION, SnapshotSeries, snapshot_states
        tracker = SnapshotSeries(cfg)
        note = {"source": "Zenodo 20046390 ccxt REST full top-100 snapshots (~1.24 s cadence)",
                "source_kind": "full_snapshots", "snapshot_adapter_version": SNAPSHOT_ADAPTER_VERSION,
                "of": "price-aware flow observed between consecutive full snapshots, accumulated before same-mid "
                      "filtering; not message-level flow",
                "timestamp_semantics": "collector timestamp of each REST depth snapshot (ccxt 'timestamp', ms)",
                "collector_warning": "REST snapshots about every 1.24 s: updates between snapshots are not observed.",
                "license": cfg.get("source_license")}
        return snapshot_states(cfg, raw, files, tracker), tracker, note
    tracker = BookReplay(cfg)
    note = {"source": "HF snapshot + sequenced absolute-quantity depth replay", "source_kind": "diff_replay",
            "of": "price-aware, accumulated before same-mid filtering",
            "timestamp_semantics": "dataset event/receipt timestamp; legacy collector does not guarantee arrival time",
            "collector_warning": "June-August 2026 may contain missing updates; IDs/timestamps cannot certify undetectable omissions."}
    return states(cfg, raw, dest / "replay_sort", tracker, files), tracker, note


def prepare(cfg):
    raw = Path(cfg["raw_dir"])
    source = json.loads((raw / "download_manifest.json").read_text())
    if source.get("status") != "complete" or source.get("provider") != cfg["provider"]:
        raise ValueError("Cần historical archive đã tải hoàn tất, đúng provider của config.")
    if source["repo"] != cfg["dataset_repo"] or source["revision"] != cfg["dataset_revision"]:
        raise ValueError("Archive revision không khớp cấu hình; không trộn data.")
    files = [f["path"] for f in source["selected_files"]]
    for name in files:
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    # Provenance of the code that runs this replay, taken before any output is written.
    provenance = code_provenance(cfg)
    dest = Path(cfg["prepared_dir"])
    dest.mkdir(parents=True, exist_ok=False)
    stream, replay, note = source_stream(cfg, raw, dest, files)
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

        for now, segment, (bp, bq, ap, aq) in stream:
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
               "source_kind": note["source_kind"], "collector_warning": note["collector_warning"]})
    write_json(dest / "manifest.json", {"schema_version": 3, "replay_version": REPLAY_VERSION, **provenance,
               "config": cfg, "historical_fixed": True,
               "dataset_repo": source["repo"], "dataset_revision": source["revision"], "coverage": coverage,
               "counts": totals, "features": feature_names(), "dtypes": DTYPES, "files": files,
               "timestamp_unit": "microseconds (source timestamp_ms * 1000)",
               "price": "L2 best-bid/best-ask mid", **note, "sequence_ids_available": True})
    print(f"Prepared observed coverage: {coverage}", flush=True)
