"""Offline snapshot + atomic Binance depth-message replay, with hard segment resets."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

# Conservative minute boundaries supplied with the public dataset.
KNOWN_GAPS = [(int(pd.Timestamp("2026-07-05T20:56:00Z").value // 1000),
               int(pd.Timestamp("2026-07-05T21:39:00Z").value // 1000))]


@dataclass
class Event:
    ts: int
    first: int
    last: int
    levels: list


def grouped_events(reader):
    """Keep one exchange message atomic across Arrow batches and source files."""
    key, levels = None, []
    for batch in reader:
        cols = batch.to_pydict()
        for ts, first, last, side, price, quantity in zip(*(cols[k] for k in
                ("timestamp_ms", "first_update_id", "last_update_id", "side", "price", "quantity"))):
            if ts is None or first is None or last is None:
                raise ValueError("Archive has rows without timestamp/sequence ids; cannot reconstruct.")
            incoming = (int(ts) * 1000, int(first), int(last))
            if key is not None and incoming != key:
                yield Event(*key, levels)
                levels = []
            key = incoming
            levels.append((side, price, quantity))
    if key is not None:
        yield Event(*key, levels)


class BookReplay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bids, self.asks = SortedDict(), SortedDict()
        self.last_id = self.last_ts = None
        self.segment = -1
        self.segments = []
        self.resets = []
        self.counts = Counter()
        self.waiting_bridge = True
        self.reset_after = -1
        self.source_first = self.source_last = None

    def observe(self, event):
        self.source_first = event.ts if self.source_first is None else min(self.source_first, event.ts)
        self.source_last = event.ts if self.source_last is None else max(self.source_last, event.ts)

    def reset(self, reason, timestamp):
        if self.last_id is not None:
            self.segments[-1]["end_exclusive_us"] = self.segments[-1]["last_us"] + 1
            self.segments[-1]["end_reason"] = reason
            self.resets.append({"timestamp_us": int(timestamp), "reason": reason, "segment": self.segment})
        self.counts[reason] += 1
        self.bids.clear()
        self.asks.clear()
        self.last_id = self.last_ts = None
        self.waiting_bridge = True
        self.reset_after = max(self.reset_after, timestamp)

    def permitted_time(self, ts):
        for start, end in KNOWN_GAPS:
            if self.last_ts is not None and self.last_ts < start <= ts:
                self.reset("known_hard_gap", start)
            if start <= ts < end:
                self.reset_after = max(self.reset_after, end - 1)
                self.counts["inside_known_gap"] += 1
                return False
        return True

    def validated_levels(self, event):
        if event.first < 0 or event.last < event.first:
            return None
        unique = {}
        for side, price, quantity in event.levels:
            if side not in ("bid", "ask") or price is None or quantity is None:
                return None
            if not np.isfinite(price) or not np.isfinite(quantity) or price <= 0 or quantity < 0:
                return None
            key = (side, float(price))
            if key in unique and unique[key] != quantity:
                return None  # conflicting rows for the same atomic update
            unique[key] = float(quantity)
        return unique

    def apply_levels(self, levels):
        for (side, price), quantity in levels.items():
            book = self.bids if side == "bid" else self.asks
            if quantity == 0:
                book.pop(price, None)
            else:
                book[price] = quantity  # absolute replacement, never += quantity
        # Prune AFTER the whole message, not after each price row or Arrow batch.
        pruned_bid = pruned_ask = False
        while len(self.bids) > self.cfg["book_max_depth"]:
            self.bids.popitem(0)
            pruned_bid = True
            self.counts["pruned_bid_levels"] += 1
        while len(self.asks) > self.cfg["book_max_depth"]:
            self.asks.popitem(-1)
            pruned_ask = True
            self.counts["pruned_ask_levels"] += 1
        # If the top 10 later retreat into forgotten depth, require a new snapshot.
        if pruned_bid:
            self.bid_floor = max(self.bid_floor, self.bids.peekitem(0)[0])
        if pruned_ask:
            self.ask_ceiling = min(self.ask_ceiling, self.asks.peekitem(-1)[0])

    def top(self):
        if min(len(self.bids), len(self.asks)) < 10:
            return None
        bp = np.asarray(list(islice(reversed(self.bids), 10)), np.float64)
        ap = np.asarray(list(islice(iter(self.asks), 10)), np.float64)
        if bp[0] >= ap[0] or bp[-1] < self.bid_floor or ap[-1] > self.ask_ceiling:
            return None
        return bp, np.asarray([self.bids[p] for p in bp]), ap, np.asarray([self.asks[p] for p in ap])

    def snapshot(self, event):
        self.observe(event)
        self.counts["snapshot_messages"] += 1
        if not self.permitted_time(event.ts) or event.ts <= self.reset_after:
            return None
        levels = self.validated_levels(event)
        if levels is None:
            self.reset("invalid_snapshot", event.ts)
            return None
        self.reset("snapshot_reanchor", event.ts)
        self.bid_floor, self.ask_ceiling = -np.inf, np.inf
        self.apply_levels(levels)
        if self.bids and self.asks:
            self.bid_floor = self.bids.peekitem(0)[0]
            self.ask_ceiling = self.asks.peekitem(-1)[0]
        top = self.top()
        if top is None:
            self.reset("invalid_snapshot_book", event.ts)
            return None
        self.segment += 1
        self.last_id, self.last_ts = event.last, event.ts
        self.segments.append({"id": self.segment, "start_us": event.ts, "last_us": event.ts,
                              "end_exclusive_us": event.ts + 1, "snapshot_update_id": event.last,
                              "states": 1, "end_reason": "archive_end"})
        return event.ts, self.segment, top

    def depth(self, event):
        self.observe(event)
        self.counts["depth_messages"] += 1
        if not self.permitted_time(event.ts):
            return None
        if self.last_id is None:
            self.counts["depth_waiting_for_snapshot"] += 1
            return None
        if event.last <= self.last_id:
            self.counts["obsolete_or_duplicate_depth"] += 1
            return None
        if event.ts < self.last_ts or event.ts - self.last_ts > self.cfg["max_feed_gap_seconds"] * 1e6:
            self.reset("invalid_timestamp_gap", max(event.ts, self.last_ts))
            return None
        # U..u must cover the next missing ID. This also permits a buffered
        # message overlapping the snapshot; fully obsolete messages were skipped.
        contiguous = event.first <= self.last_id + 1 <= event.last
        if not contiguous:
            self.reset("sequence_gap", event.ts)
            return None
        if event.first <= self.last_id:
            self.counts["overlapping_contiguous_messages"] += 1
        levels = self.validated_levels(event)
        if levels is None:
            self.reset("invalid_depth_message", event.ts)
            return None
        self.apply_levels(levels)
        top = self.top()
        if top is None:
            self.reset("crossed_insufficient_or_unknown_depth", event.ts)
            return None
        self.last_id, self.last_ts = event.last, event.ts
        self.waiting_bridge = False
        self.segments[-1].update(last_us=event.ts, end_exclusive_us=event.ts + 1)
        self.segments[-1]["states"] += 1
        return event.ts, self.segment, top


def states(cfg, raw, scratch, replay, source_files):
    """External sort bounds RAM; IDs drive depth order, timestamp inversions reset.

    Snapshot anchors are consumed only when their timestamp is observable. Never
    initialize an earlier depth event using a later snapshot.
    """
    import duckdb

    scratch.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(config={"memory_limit": "1GB", "temp_directory": str(scratch), "threads": 2})
    try:
        def query(kind):
            paths = [str(raw / p) for p in source_files if p.startswith(kind + "/")]
            first = "last_update_id" if kind == "snapshots" else "first_update_id"
            order = "timestamp_ms, last_update_id" if kind == "snapshots" else "last_update_id, first_update_id, timestamp_ms"
            sql = (f"SELECT timestamp_ms, {first} AS first_update_id, last_update_id, side, price, quantity "
                   f"FROM read_parquet(?) WHERE exchange = ? AND asset = ? ORDER BY {order}, side, price")
            return connection.execute(sql, [paths, cfg["exchange"], cfg["symbol"]]).fetch_record_batch(cfg["chunk_rows"])

        snapshots = iter(list(grouped_events(query("snapshots"))))
        anchor = next(snapshots, None)
        last_depth_ts = -1
        for event in grouped_events(query("depth")):
            if event.ts < last_depth_ts:
                replay.observe(event)
                replay.reset("depth_timestamp_reversal", last_depth_ts)
                continue
            last_depth_ts = event.ts
            while anchor is not None and anchor.ts <= event.ts:
                value = replay.snapshot(anchor)
                if value is not None:
                    yield value
                anchor = next(snapshots, None)
            value = replay.depth(event)
            if value is not None:
                yield value
        # No synthetic continuation through snapshots beyond the depth archive.
    finally:
        connection.close()
