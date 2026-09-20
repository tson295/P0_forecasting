"""Second-gate contracts: opt-in row repair, 10s cadence, price-space metrics."""
import numpy as np
import pandas as pd
import pytest

from src.config import Config, DataConfig
from src.data.dataset import prepare_data
from src.data.preprocessing import load_csv
from src.models import build_model
from src.utils.metrics import HORIZON_LABELS, metrics_from_arrays, price_metrics
from src.utils.predictions import frame_metrics, prediction_frame
from tests.test_data import write_csv

START_US = 1_758_067_200_000_000  # 2025-09-17T00:00:00Z, the first row of the real file.
STEP_US = 10_000_000
# Same capacity contract as tests/test_contracts.py: history_seconds 490 must re-resolve
# to the history_rows=49 these counts were frozen at on the 1.25s Oct-2023 file.
CAPACITY = dict(ofi_lstm=(20, 55491), hfformer=(38, 22026), patchtst=(40, 477059),
                moderntcn=(40, 50568195), lit=(40, 736547))
PRICE_KEYS = {"samples", "horizons", "units", "mse", "rmse", "mae", "r2",
              "rmse_e0", "rmse_gain_vs_e0", "mae_e0", "mean_mse"}


def write_us_csv(path, stamps_us, key=None, segments=None):
    """timestamp_us variant of tests.test_data.write_csv, orderable row by row.

    bid_price_1 rises strictly with `key`, which defaults to the timestamp, so a
    permutation applied to the timestamps but not to the book (or applied twice)
    shows up as a price that no longer matches its own row.
    """
    stamps = np.asarray(stamps_us, dtype=np.int64)
    key = (stamps-stamps.min())/1e6 if key is None else np.asarray(key, dtype=np.float64)
    frame = pd.DataFrame({"timestamp_us": stamps})
    if segments is not None:
        frame["segment_id"] = segments
    for side in ("bid", "ask"):
        for level in range(1, 11):
            frame[f"{side}_price_{level}"] = 100+key*1e-3+(-level if side == "bid" else level)*.01
            frame[f"{side}_qty_{level}"] = level+key*1e-4
    frame.to_csv(path, index=False)
    return frame


def write_grid(path, rows=1200):
    """The perfect 10s grid the real file sits on."""
    return write_us_csv(path, START_US+np.arange(rows)*STEP_US)


def us_config(path, **kwargs):
    return DataConfig(csv_path=str(path), timestamp_column="timestamp_us",
                      timestamp_unit="us", **kwargs)


def gate_config(path, model="e0", **kwargs):
    config = Config(model=model)
    config.data = us_config(path, **(dict(history_seconds=490, max_gap_seconds=10.,
                                          target_tolerance_seconds=10.) | kwargs))
    return config


@pytest.fixture(scope="module")
def gate_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("gate1y")/"grid.csv"
    write_grid(path)
    return path


def test_out_of_order_blocks_need_an_explicit_opt_in(tmp_path):
    path = tmp_path/"blocks.csv"
    # The real file prepends a later block in front of the main one; same shape here.
    write_us_csv(path, np.r_[START_US+np.arange(200, 220)*STEP_US, START_US+np.arange(100)*STEP_US])
    defaults = DataConfig()
    assert defaults.sort_by_timestamp is False and defaults.duplicate_timestamp_policy == "error"
    with pytest.raises(ValueError, match="Timestamps must be strictly increasing"):
        load_csv(us_config(path))
    raw = load_csv(us_config(path, sort_by_timestamp=True))
    repairs = raw.source["row_repairs"]
    assert repairs["sorted_by_timestamp"] is True and repairs["rows_moved_by_sort"] == 120
    assert repairs["rows_in_file"] == repairs["rows_used"] == 120
    assert repairs["duplicate_rows_dropped"] == 0
    assert (np.diff(raw.timestamps) > 0).all()


def test_duplicate_timestamp_policy_picks_a_payload(tmp_path):
    path = tmp_path/"dupes.csv"
    stamps = START_US+np.arange(60)*STEP_US
    stamps[31] = stamps[30]  # One timestamp, two snapshots that disagree.
    key = np.arange(60, dtype=np.float64)
    key[30], key[31] = 1000., 2000.
    write_us_csv(path, stamps, key)
    with pytest.raises(ValueError, match="Timestamps must be strictly increasing"):
        load_csv(us_config(path))
    for policy, kept in (("keep_first", 1000.), ("keep_last", 2000.)):
        raw = load_csv(us_config(path, duplicate_timestamp_policy=policy))
        assert raw.source["row_repairs"]["duplicate_rows_dropped"] == 1
        assert len(raw.timestamps) == 59 and (np.diff(raw.timestamps) > 0).all()
        # The row count cannot tell the two policies apart; the surviving payload can.
        assert raw.book[30, 0, 0, 0] == 100+kept*1e-3-.01


def test_sorting_permutes_book_and_segments_with_the_timestamps(tmp_path):
    path = tmp_path/"permuted.csv"
    stamps = np.r_[START_US+np.arange(200, 220)*STEP_US, START_US+np.arange(100)*STEP_US]
    write_us_csv(path, stamps, segments=np.r_[np.full(20, "s1"), np.full(100, "s0")])
    raw = load_csv(us_config(path, sort_by_timestamp=True))
    elapsed = (raw.timestamps-raw.timestamps.min())/1e9
    # Row i must still carry the book that arrived with timestamp i.
    np.testing.assert_allclose(raw.book[:, 0, 0, 0], 100+elapsed*1e-3-.01, rtol=0, atol=1e-12)
    assert (np.diff(raw.book[:, 0, 0, 0]) > 0).all()
    # factorize runs before the sort, in file order, so the later block stays code 0.
    np.testing.assert_array_equal(raw.segments, np.r_[np.ones(100), np.zeros(20)])
    assert raw.stats["segment_transitions"] == 1


def test_timestamp_us_keeps_microsecond_resolution(tmp_path):
    path = tmp_path/"us.csv"
    stamps = START_US+np.arange(40)*STEP_US
    write_us_csv(path, stamps)
    raw = load_csv(us_config(path))
    assert raw.timestamps[0] == START_US*1000
    assert set(np.diff(raw.timestamps).tolist()) == {10_000_000_000}
    stamps[5] += 1  # A single microsecond: millisecond parsing would swallow it.
    write_us_csv(path, stamps)
    assert np.diff(load_csv(us_config(path)).timestamps)[4] == 10_000_001_000


def test_gap_rule_scales_with_cadence(tmp_path):
    # The frozen 2.0s rule still fits the 1.25s Oct-2023 file...
    fast = tmp_path/"fast.csv"
    write_csv(fast)
    old = Config(model="e0")
    old.data.csv_path = str(fast)
    assert prepare_data(old).history_rows == 48
    # ...but on a 10s grid every single edge is longer than it, so nothing survives.
    slow = tmp_path/"slow.csv"
    write_grid(slow)
    raw = load_csv(us_config(slow, max_gap_seconds=2.))
    assert raw.bad_edges.all() and raw.stats["gaps_gt_max_gap"] == len(raw.timestamps)-1
    with pytest.raises(ValueError, match="no continuous sampling intervals"):
        prepare_data(gate_config(slow, max_gap_seconds=2., target_tolerance_seconds=2.))
    data = prepare_data(gate_config(slow))
    assert load_csv(us_config(slow, max_gap_seconds=10.)).stats["gaps_gt_max_gap"] == 0
    assert data.history_rows == 49 and data.metadata["preprocessing"]["stride_rows"] == 8
    assert all(len(dataset) for dataset in data.datasets.values())


def test_price_metric_contract_for_perfect_and_e0_predictions():
    rng = np.random.default_rng(0)
    # Real mid range and real E0 error scale, so the R2 clause below is not a toy.
    origin = np.repeat(rng.uniform(57_000, 126_000, size=(512, 1)), 3, axis=1)
    target = origin+rng.normal(scale=[50.4, 71.7, 87.9], size=(512, 3))
    perfect, e0 = price_metrics(origin, target, target), price_metrics(origin, target, origin)
    assert perfect["rmse"] == [0., 0., 0.] and perfect["rmse_gain_vs_e0"] == [1., 1., 1.]
    # E0's predicted mid IS the origin mid, so it must score exactly its own baseline.
    assert e0["rmse"] == e0["rmse_e0"] and e0["rmse_gain_vs_e0"] == [0., 0., 0.]
    np.testing.assert_allclose(e0["rmse_e0"], np.sqrt(((target-origin)**2).mean(axis=0)),
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(e0["mae_e0"], np.abs(target-origin).mean(axis=0), rtol=1e-12, atol=0)
    # Known and deliberate: sigma(mid) is five figures and every error is two, so
    # price-space R2 is ~1 even for E0. Read rmse_gain_vs_e0, do not "fix" R2.
    assert min(e0["r2"]) > .999
    result = price_metrics(origin, target, origin+rng.normal(scale=40., size=(512, 3)))
    assert set(result) == PRICE_KEYS and result["units"] == "quote_currency"
    assert result["horizons"] == list(HORIZON_LABELS) and result["samples"] == 512
    assert all(len(result[k]) == 3 for k in ("rmse", "mae", "r2", "rmse_e0", "rmse_gain_vs_e0"))
    assert result["mean_mse"] == pytest.approx(float(np.mean(result["mse"])), rel=1e-12)
    with pytest.raises(ValueError, match="share a shape"):
        price_metrics(origin, target, origin[:-1])
    for bad in (np.zeros((0, 3)), np.zeros((8, 2)), np.zeros(8)):
        with pytest.raises(ValueError, match=r"non-empty \[N, 3\]"):
            price_metrics(bad, bad, bad)


def test_frame_metrics_are_price_units_over_a_log_return_block(gate_csv):
    data = prepare_data(gate_config(gate_csv))
    dataset, raw = data.datasets["test"], data.raw
    truth = np.log(raw.mid[dataset.target_indices]/raw.mid[dataset.origins][:, None])
    for predictions, gain in ((truth, 1.), (np.zeros_like(truth), 0.)):
        metrics = frame_metrics(prediction_frame(dataset, raw, predictions))
        assert metrics["units"] == "quote_currency" and "units" not in metrics["log_return"]
        assert metrics["log_return"]["rmse_gain_vs_e0"] == [gain]*3
        # Price space round-trips through exp(log(.)), so it agrees to float noise only.
        assert metrics["rmse_gain_vs_e0"] == pytest.approx([gain]*3, abs=1e-9)
    frame = prediction_frame(dataset, raw,
                             np.random.default_rng(1).normal(scale=1e-4, size=(len(dataset), 3)))
    metrics = frame_metrics(frame)
    columns = lambda prefix: frame[[f"{prefix}_{h}" for h in HORIZON_LABELS]].to_numpy()
    origin = frame[["origin_mid"]].to_numpy()
    np.testing.assert_allclose(metrics["rmse_e0"],
                               np.sqrt(((columns("target_mid")-origin)**2).mean(axis=0)),
                               rtol=1e-12, atol=0)
    assert metrics["log_return"] == metrics_from_arrays(columns("pred_return"), columns("true_return"))
    # A mid near 100 makes every price figure ~100x its log-return counterpart.
    assert metrics["rmse"][0] > 50*metrics["log_return"]["rmse"][0]


@pytest.mark.parametrize("name", CAPACITY)
def test_history_seconds_490_keeps_the_frozen_capacity(gate_csv, name):
    channels, parameters = CAPACITY[name]
    data = prepare_data(gate_config(gate_csv, model=name))
    assert data.history_rows == 49 and data.channels == channels
    model = build_model(name, data.history_rows, data.channels)
    assert sum(p.numel() for p in model.parameters()) == parameters
