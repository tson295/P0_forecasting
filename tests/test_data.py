import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import Config, DataConfig
from src.data.dataset import LOBDataset, prepare_data
from src.data.ofi import order_flow
from src.data.preprocessing import RAW_COLUMNS, features, load_csv


def raw_timeline(ts, segments=None):
    ts = np.rint(np.asarray(ts)*1e9).astype(np.int64)
    segments = np.zeros(len(ts)) if segments is None else np.asarray(segments)
    bad = (np.diff(ts) > 2e9) | (segments[1:] != segments[:-1])
    return SimpleNamespace(timestamps=ts, mid=100+np.arange(len(ts))*.01,
                           bad_prefix=np.r_[0, np.cumsum(bad)])


def test_irregular_lookup_gap_segment_and_split():
    ts = np.cumsum(np.resize([1.12, 1.3, 1.4, 1.24], 1200))
    ts[300:] += 3.0
    segments = np.zeros(len(ts)); segments[700:] = 1
    raw = raw_timeline(ts, segments)
    c = DataConfig()
    bounds = (20, 1150)
    ds = LOBDataset(np.zeros((len(ts), 40), np.float32), raw, bounds, 49, 8, c)
    # Independent slow oracle checks EVERY candidate, including rejected candidates.
    expected = []
    for origin in range(20+48, 1150, 8):
        wanted = raw.timestamps[origin]+np.array([60, 120, 180])*10**9
        target = np.searchsorted(raw.timestamps, wanted)
        if target[-1] >= bounds[1]:
            continue
        if np.any(raw.timestamps[target]-wanted > 2e9):
            continue
        start = origin-48
        if np.any(np.diff(raw.timestamps[start:target[-1]+1]) > 2e9):
            continue
        if np.any(segments[start:target[-1]+1] != segments[start]):
            continue
        expected.append(origin)
    assert ds.origins.tolist() == expected
    assert len(ds) > 0
    for i in (0, len(ds)-1):
        x, y = ds[i]
        assert x.shape == (49, 40) and y.shape == (3,)
        assert np.shares_memory(x.numpy(), ds.x)
        wanted = raw.timestamps[ds.origins[i]]+np.array([60, 120, 180])*10**9
        assert np.all(raw.timestamps[ds.target_indices[i]-1] < wanted)


def test_exact_two_second_edges_are_allowed_and_tolerance_inclusive():
    ts = np.arange(400)*2.0
    raw = raw_timeline(ts)
    c = DataConfig(target_tolerance_seconds=0)
    ds = LOBDataset(np.zeros((400, 1), np.float32), raw, (0, 400), 3, 1, c)
    assert len(ds) == 308
    assert ds.origins[0] == 2
    c.horizons_seconds = (61, 121, 181)
    c.target_tolerance_seconds = 1
    ds = LOBDataset(np.zeros((400, 1), np.float32), raw, (0, 400), 3, 1, c)
    assert len(ds) > 0
    c.target_tolerance_seconds = .999
    assert len(LOBDataset(ds.x, raw, (0, 400), 3, 1, c)) == 0


def test_of_signs_levels_and_boundary_reset():
    book = np.zeros((4, 2, 10, 2))
    book[..., 1] = np.array([10, 12, 7, 8])[:, None, None]
    book[:, 0, :, 0] = np.array([100, 101, 101, 100])[:, None]
    book[:, 1, :, 0] = np.array([102, 101, 101, 102])[:, None]
    reset = np.array([True, False, False, False])
    of = order_flow(book, reset)
    np.testing.assert_array_equal(of[:, 0], [0, 12, -5, -7])
    np.testing.assert_array_equal(of[:, 10], [0, 12, -5, -7])
    np.testing.assert_array_equal(order_flow(book, reset, "ofi"), 0)
    reset[1] = True
    assert (order_flow(book, reset)[1] == 0).all()


def write_csv(path, n=3000):
    frame = pd.DataFrame({"timestamp_ms": 1696118400000+np.arange(n)*1250,
                          "segment_id": np.zeros(n, dtype=int)})
    for side in ("bid", "ask"):
        for level in range(1, 11):
            price = 100+np.arange(n)*.001+(-level if side == "bid" else level)*.01
            frame[f"{side}_price_{level}"] = price
            frame[f"{side}_qty_{level}"] = level+np.arange(n)*.0001
    frame.to_csv(path, index=False)
    return frame


def test_train_only_scaling_and_saved_preprocessing(tmp_path):
    path = tmp_path/"data.csv"
    frame = write_csv(path)
    c = Config(model="hfformer"); c.data.csv_path = str(path)
    a = prepare_data(c)
    hi = a.metadata["split_manifest"]["ranges"]["train"][1]
    frame.loc[hi:, [k for k in RAW_COLUMNS if "price" in k]] *= 10
    frame.to_csv(path, index=False)
    b = prepare_data(c)
    assert a.metadata["preprocessing"]["standardizer"] == b.metadata["preprocessing"]["standardizer"]
    assert a.history_rows == b.history_rows == 48
    assert a.metadata["preprocessing"]["stride_rows"] == 8
    np.testing.assert_array_equal(a.datasets["train"][0][0], b.datasets["train"][0][0])
    # Reusing a checkpoint does not refit even when the new TRAIN data changes.
    frame.loc[:, [k for k in RAW_COLUMNS if "qty" in k]] *= 5
    frame.to_csv(path, index=False)
    restored = prepare_data(c, a.metadata)
    assert restored.metadata["preprocessing"]["standardizer"] == a.metadata["preprocessing"]["standardizer"]


def test_mapping_precision_and_invalid_order(tmp_path):
    path = tmp_path/"data.csv"
    frame = write_csv(path)
    frame.rename(columns={"bid_price_1": "best_bid"}).to_csv(path, index=False)
    c = DataConfig(csv_path=str(path), column_mapping={"bid_price_1": "best_bid"})
    raw = load_csv(c)
    assert raw.timestamps[1]-raw.timestamps[0] == 1_250_000_000
    assert raw.column_mapping["features"]["bid_price_1"] == "best_bid"
    assert raw.book.dtype == np.float64
    frame.loc[1, "timestamp_ms"] = frame.loc[0, "timestamp_ms"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_csv(DataConfig(csv_path=str(path)))


def test_feature_contracts_and_weighted_mid(tmp_path):
    path = tmp_path/"data.csv"; write_csv(path)
    raw = load_csv(DataConfig(csv_path=str(path)))
    ranges = dict(train=(0, 2000), validation=(2000, 2500), test=(2500, 3000))
    raw.book[0, 0, 0, 1], raw.book[0, 1, 0, 1] = 3., 1.
    hf, schema = features(raw, "hfformer", "of", ranges)
    assert hf.shape == (3000, 38)
    assert hf[0, -1] == pytest.approx((3*raw.book[0, 1, 0, 0]+raw.book[0, 0, 0, 0])/4)
    assert (hf[[0, 2000, 2500], -2] == 0).all()
    assert not any(name.endswith("_10") for name in schema["names"])
    lit, _ = features(raw, "lit", "of", ranges)
    assert lit.shape == (3000, 2, 10, 2)
    np.testing.assert_array_equal(lit, raw.book)
    of, _ = features(raw, "ofi_lstm", "of", ranges)
    assert (of[[0, 2000, 2500]] == 0).all()
