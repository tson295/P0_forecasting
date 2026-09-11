"""Full order-book snapshot series (Zenodo 20046390: ccxt REST JSON lines) as prepared book states.

Each line is a complete top-100 book, so no diff replay is needed: every valid snapshot is a state.
A segment ends on a timestamp gap > max_feed_gap_seconds, a timestamp or nonce (Binance lastUpdateId)
that does not increase, or an invalid book/line; the next valid snapshot starts a new segment
(no forward fill, no joining). OF is the flow observed between consecutive snapshots, not message flow.
"""
from __future__ import annotations

import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

SNAPSHOT_ADAPTER_VERSION = 1
TAIL = re.compile(rb'"timestamp":\s*(\d+),\s*"datetime":\s*[^,]*,\s*"nonce":\s*(\d+)')


class SnapshotSeries:
    """Segment bookkeeping with the same attributes prepare() reads from BookReplay."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.window = int(cfg["max_feed_gap_seconds"] * 1e6)
        self.segment = -1
        self.segments, self.resets = [], []
        self.counts = Counter()
        self.known_gaps = []  # an independent source declares no hard gaps
        self.last_ts = None  # last state of the open segment
        self.emitted = None  # (ts, nonce) of the last emitted state; the raw timeline never moves backwards
        self.source_first = self.source_last = None

    def observe(self, ts):
        self.source_first = ts if self.source_first is None else min(self.source_first, ts)
        self.source_last = ts if self.source_last is None else max(self.source_last, ts)

    def reset(self, reason, timestamp):
        if self.last_ts is not None:
            self.segments[-1]["end_exclusive_us"] = self.segments[-1]["last_us"] + 1
            self.segments[-1]["end_reason"] = reason
            self.resets.append({"timestamp_us": int(timestamp), "reason": reason, "segment": self.segment})
        self.counts[reason] += 1
        self.last_ts = None

    @staticmethod
    def top(book):
        """Top 10 per side of a full snapshot, or None unless sorted, positive, finite and not crossed."""
        try:
            bids = np.asarray(book["bids"][:10], np.float64)
            asks = np.asarray(book["asks"][:10], np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        if bids.shape != (10, 2) or asks.shape != (10, 2):
            return None
        if not (np.isfinite(bids).all() and np.isfinite(asks).all()) or (bids <= 0).any() or (asks <= 0).any():
            return None
        if not (np.all(np.diff(bids[:, 0]) < 0) and np.all(np.diff(asks[:, 0]) > 0) and bids[0, 0] < asks[0, 0]):
            return None
        return bids[:, 0], bids[:, 1], asks[:, 0], asks[:, 1]

    def invalid(self, ts):
        """A line that is not a parseable snapshot closes the segment; it is never repaired."""
        self.counts["snapshot_messages"] += 1
        if ts is not None:
            self.observe(ts)
        self.reset("invalid_snapshot_line", ts if ts is not None else (self.last_ts or 0))

    def snapshot(self, ts, nonce, book):
        self.observe(ts)
        self.counts["snapshot_messages"] += 1
        if self.emitted is not None and (ts <= self.emitted[0] or nonce <= self.emitted[1]):
            # Out-of-order or repeated response: close the segment and drop it (timeline stays monotonic).
            self.reset("timestamp_not_increasing" if ts <= self.emitted[0] else "nonce_not_increasing", ts)
            self.counts["snapshots_dropped_out_of_order"] += 1
            return None
        if self.last_ts is not None and ts - self.last_ts > self.window:
            self.reset("invalid_timestamp_gap", ts)
        top = self.top(book)
        if top is None:
            self.reset("invalid_snapshot_book", ts)
            return None
        if self.last_ts is None:
            self.segment += 1
            self.segments.append({"id": self.segment, "start_us": ts, "last_us": ts, "end_exclusive_us": ts + 1,
                                  "snapshot_update_id": nonce, "states": 0, "end_reason": "archive_end"})
        self.last_ts = ts
        self.emitted = (ts, nonce)
        self.segments[-1].update(last_us=ts, end_exclusive_us=ts + 1)
        self.segments[-1]["states"] += 1
        return ts, self.segment, top


def snapshot_states(cfg, raw, source_files, series):
    """Stream daily JSON-lines members of the pinned tar.gz in chronological order."""
    for name in source_files:
        with tarfile.open(Path(raw) / name) as archive:
            # Skip AppleDouble "._*" entries; daily files are BTCUSDT_MM-DD.json within one month.
            members = sorted((m for m in archive.getmembers() if m.isfile() and m.name.endswith(".json")
                              and Path(m.name).name.startswith("BTCUSDT_")), key=lambda m: Path(m.name).name)
            for member in members:
                for line in archive.extractfile(member):
                    if not line.strip():
                        continue
                    try:
                        book = json.loads(line)
                        if book.get("symbol") != "BTC/USDT":
                            raise ValueError("unexpected symbol")
                        ts, nonce = int(book["timestamp"]) * 1000, int(book["nonce"])
                    except (ValueError, KeyError, TypeError):
                        tail = TAIL.search(line[-200:])
                        series.invalid(int(tail.group(1)) * 1000 if tail else None)
                        continue
                    value = series.snapshot(ts, nonce, book)
                    if value is not None:
                        yield value
