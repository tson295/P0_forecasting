"""Third-gate contracts: 3-fold walk forward, purge+embargo, R2 gain vs E0, scaled capacity."""
import gc

import numpy as np
import pytest

from src.config import Config, DataConfig
from src.data.dataset import LOBDataset, prepare_data
from src.data.preprocessing import (chronological_split, load_csv, target_limits,
                                    walk_forward_split)
from src.models import build_model
from src.utils.metrics import HORIZON_LABELS, price_metrics
from src.utils.predictions import frame_metrics, prediction_frame
from tests.test_gate1y import gate_config, us_config, write_grid

# A 10s grid small enough to stay in milliseconds and large enough to reproduce the real
# file's geometry: 0.85 of the span cut into four 425-row blocks, then a 300-row test tail.
ROWS, BLOCK_ROWS, TEST_LO = 2000, 425, 1700
FOLDS, TEST_FRACTION, EMBARGO = 3, .15, 180.
HISTORY_ROWS, STRIDE_ROWS = 49, 2  # ceil(490/10) and round(20/10) on this cadence.
HORIZON_ROWS = 18                  # The 180s horizon, in rows.
# price_metrics no longer reports mean-based R2 on price; r2_gain_vs_e0 replaced it.
PRICE_KEYS = {"samples", "horizons", "units", "mse", "rmse", "mae", "r2_gain_vs_e0",
              "rmse_e0", "rmse_gain_vs_e0", "mae_e0", "mean_mse"}
CAPACITY = dict(ofi_lstm=(20, 2_990_211), hfformer=(38, 4_677_278),
                patchtst=(40, 14_434_563), moderntcn=(40, 50_568_195), lit=(40, 15_995_235))


def wf_config(path, fold=1, embargo=EMBARGO, model="e0", **kwargs):
    """configs/{model}_wf3_f{fold}.json in miniature, pointed at a synthetic grid."""
    return gate_config(path, model=model, stride_seconds=20., split_scheme="walk_forward",
                       folds=FOLDS, fold=fold, test_fraction=TEST_FRACTION,
                       embargo_seconds=embargo, **kwargs)


def touched(dataset, history_rows=HISTORY_ROWS):
    """First raw row the samples read, last raw row they read: history start .. last target."""
    return int((dataset.origins-history_rows+1).min()), int(dataset.target_indices.max())


@pytest.fixture(scope="module")
def grid_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("wf3")/"grid.csv"
    write_grid(path, rows=ROWS)
    return path


@pytest.fixture(scope="module")
def raw(grid_csv):
    return load_csv(us_config(grid_csv, max_gap_seconds=10.))


@pytest.fixture(scope="module")
def prepared(grid_csv):
    """Memoized prepare_data: every fold/embargo pair is read once, not once per assertion."""
    cache = {}

    def get(fold=1, embargo=EMBARGO):
        if (fold, embargo) not in cache:
            cache[fold, embargo] = prepare_data(wf_config(grid_csv, fold, embargo))
        return cache[fold, embargo]
    return get


def test_fold_geometry_expands_and_holds_one_test_tail(raw, grid_csv):
    ts = raw.timestamps
    folds = {k: walk_forward_split(raw, wf_config(grid_csv, k).data) for k in (1, 2, 3)}
    ranges = {k: r for k, (r, _) in folds.items()}
    for k, r in ranges.items():
        manifest = folds[k][1]
        assert manifest["scheme"] == "walk_forward" and manifest["fold"] == k
        assert (manifest["folds"], manifest["embargo_seconds"]) == (FOLDS, EMBARGO)
        # Train is blocks [0, k), validation is block k, test is the tail.
        assert r["train"] == (0, k*BLOCK_ROWS)
        assert r["validation"] == (k*BLOCK_ROWS, (k+1)*BLOCK_ROWS)
        assert r["test"] == (TEST_LO, ROWS)
    # Expanding window: fold k+1 trains on exactly what fold k trained AND validated on.
    for k in (1, 2):
        assert ranges[k+1]["train"] == (0, ranges[k]["validation"][1])
    train_hi = [ranges[k]["train"][1] for k in (1, 2, 3)]
    assert train_hi == sorted(set(train_hi)) and max(train_hi) < TEST_LO
    # One held-out tail, identical for every fold, and it is the last test_fraction of time.
    assert len({ranges[k]["test"] for k in (1, 2, 3)}) == 1
    duration = int(ts[-1])-int(ts[0])
    assert (int(ts[TEST_LO])-int(ts[0]))/duration == pytest.approx(1-TEST_FRACTION, abs=1/ROWS)
    # No fold validates on a row another fold validates on.
    for a, b in ((1, 2), (1, 3), (2, 3)):
        lo_a, hi_a = ranges[a]["validation"]
        lo_b, hi_b = ranges[b]["validation"]
        assert hi_a <= lo_b or hi_b <= lo_a
    # The last fold's validation abuts the tail, so nothing between the blocks is thrown away.
    assert ranges[3]["validation"][1] == TEST_LO


def test_embargo_separates_the_last_train_target_from_the_first_validation_row(prepared, raw):
    ts = raw.timestamps
    for fold in (1, 2, 3):
        separation = {}
        for embargo in (EMBARGO, 0.):
            data = prepared(fold, embargo)
            assert data.history_rows == HISTORY_ROWS
            assert data.metadata["preprocessing"]["stride_rows"] == STRIDE_ROWS
            ranges = data.metadata["split_manifest"]["ranges"]
            for name, dataset in data.datasets.items():
                lo, hi = ranges[name]
                first, last = touched(dataset)
                # Purging, embargo or not: history, origin and every target stay in one split.
                assert lo <= first and last < hi
            train, validation = data.datasets["train"], data.datasets["validation"]
            separation[embargo] = (int(ts[touched(validation)[0]])
                                   - int(ts[touched(train)[1]]))/1e9
        # The embargo has to outlast the 180s horizon, exactly as the real folds do at 190-200s.
        assert separation[EMBARGO] > EMBARGO
        # ...and without it the same measurement falls under the bar, so this test has teeth.
        assert separation[0.] < EMBARGO


def test_embargo_drops_exactly_the_samples_that_reach_into_the_blackout(prepared, raw):
    ts = raw.timestamps
    for fold in (1, 2, 3):
        held, free = (prepared(fold, e).datasets["train"] for e in (EMBARGO, 0.))
        assert len(held) < len(free)
        limit = int(ts[prepared(fold).metadata["split_manifest"]["ranges"]["validation"][0]])
        limit -= round(EMBARGO*1e9)
        keep = ts[free.target_indices[:, -1]] < limit
        assert not keep.all()  # Something was actually in the blackout to drop.
        np.testing.assert_array_equal(held.origins, free.origins[keep])
        assert set(free.origins.tolist())-set(held.origins.tolist()) == set(
            free.origins[~keep].tolist())


def test_target_limit_is_applied_not_just_recorded(raw, grid_csv):
    config = us_config(grid_csv, max_gap_seconds=10., target_tolerance_seconds=10.)
    x = np.zeros((len(raw.timestamps), 1), np.float32)
    limit = int(raw.timestamps[900])
    free = LOBDataset(x, raw, (0, 1200), HISTORY_ROWS, STRIDE_ROWS, config)
    held = LOBDataset(x, raw, (0, 1200), HISTORY_ROWS, STRIDE_ROWS, config, target_limit_ns=limit)
    last_free = raw.timestamps[free.target_indices[:, -1]]
    assert (last_free >= limit).any()  # Unlimited, samples do cross the bound...
    assert free.target_indices.max() < 1200  # ...but never past the split's own last row.
    assert len(held) and (raw.timestamps[held.target_indices[:, -1]] < limit).all()
    # Targets rise with the horizon, so bounding the last one bounds all three.
    assert (raw.timestamps[held.target_indices] < limit).all()
    np.testing.assert_array_equal(held.origins, free.origins[last_free < limit])


def test_no_fold_lets_train_and_validation_share_a_row(prepared, raw):
    for fold in (1, 2, 3):
        data = prepared(fold)
        train, validation, test = (data.datasets[n] for n in ("train", "validation", "test"))
        for earlier, later in ((train, validation), (validation, test), (train, test)):
            assert not set(earlier.origins.tolist()) & set(later.origins.tolist())
            # Not just the origins: the whole row span, history through last target.
            assert touched(earlier)[1] < touched(later)[0]
        assert len(train) and len(validation) and len(test)


def test_price_r2_is_gain_vs_e0_and_mean_based_r2_is_gone(prepared, raw):
    rng = np.random.default_rng(0)
    # Real mid range and real E0 error scale: the corners below are not a toy.
    origin = np.repeat(rng.uniform(57_000, 126_000, size=(512, 1)), 3, axis=1)
    target = origin+rng.normal(scale=[50.4, 71.7, 87.9], size=(512, 3))
    # E0 predicts the origin mid, so its SSE is the baseline SSE: gain is exactly zero.
    assert price_metrics(origin, target, origin)["r2_gain_vs_e0"] == [0., 0., 0.]
    assert price_metrics(origin, target, target)["r2_gain_vs_e0"] == [1., 1., 1.]
    # Halving E0's error quarters its SSE, so the gain is 1-1/4.
    half = price_metrics(origin, target, origin+(target-origin)/2)
    assert half["r2_gain_vs_e0"] == pytest.approx([.75]*3, rel=1e-12)
    assert half["rmse_gain_vs_e0"] == pytest.approx([.5]*3, rel=1e-12)
    result = price_metrics(origin, target, origin+rng.normal(scale=40., size=(512, 3)))
    assert set(result) == PRICE_KEYS and "r2" not in result
    # The squared-error twin of the RMSE gain, away from both collapsed corners.
    assert result["r2_gain_vs_e0"] == pytest.approx(
        [1-(1-g)**2 for g in result["rmse_gain_vs_e0"]], rel=1e-12)
    np.testing.assert_allclose(result["r2_gain_vs_e0"],
                               1-np.asarray(result["mse"])*len(target)
                               / ((origin-target)**2).sum(axis=0), rtol=1e-12, atol=0)
    assert max(result["r2_gain_vs_e0"]) < 0  # Worse than E0 must read negative.
    # The exported tables carry the same contract, over a log-return block that keeps its R2.
    data = prepared()
    dataset, book = data.datasets["test"], data.raw
    truth = np.log(book.mid[dataset.target_indices]/book.mid[dataset.origins][:, None])
    for predictions, gain in ((np.zeros_like(truth), 0.), (truth, 1.)):
        metrics = frame_metrics(prediction_frame(dataset, book, predictions))
        assert "r2" not in metrics and "r2_gain_vs_e0" not in metrics["log_return"]
        assert metrics["r2_gain_vs_e0"] == pytest.approx([gain]*3, abs=1e-9)
        # The nested block still reports mean-based R2, which is meaningful on returns.
        assert len(metrics["log_return"]["r2"]) == len(HORIZON_LABELS)
        assert all(r is not None for r in metrics["log_return"]["r2"])


def test_scaled_model_kwargs_hit_the_frozen_capacities(grid_csv):
    for name, (channels, parameters) in CAPACITY.items():
        config = Config.load(f"configs/{name}_wf3_f1.json")
        data = prepare_data(wf_config(grid_csv, model=name))
        assert data.history_rows == HISTORY_ROWS and data.channels == channels
        model = build_model(name, data.history_rows, data.channels, **config.model_kwargs)
        assert sum(p.numel() for p in model.parameters()) == parameters
        # 50M parameters for ModernTCN alone: release each one before building the next.
        del model, data
        gc.collect()


def test_chronological_split_is_untouched(raw, grid_csv):
    defaults = DataConfig()
    assert defaults.split_scheme == "chronological" and defaults.embargo_seconds == 0.
    assert (defaults.folds, defaults.fold, defaults.test_fraction) == (FOLDS, 1, TEST_FRACTION)
    config = gate_config(grid_csv)  # No walk-forward keys at all: the previous suites' shape.
    assert config.data.split_scheme == "chronological"
    data = prepare_data(config)
    manifest = data.metadata["split_manifest"]
    assert manifest["policy"].startswith("strict:") and "scheme" not in manifest
    assert "fold" not in manifest and "block_boundaries_ns" not in manifest
    ranges, _ = chronological_split(raw, config.data)
    assert manifest["ranges"] == ranges
    assert ranges == dict(train=(0, 1400), validation=(1400, TEST_LO), test=(TEST_LO, ROWS))
    # embargo_seconds 0: each split is purged up to the next split's first record, no further.
    limits = target_limits(raw, ranges, config.data)
    assert limits == dict(train=int(raw.timestamps[ranges["validation"][0]]),
                          validation=int(raw.timestamps[ranges["test"][0]]), test=None)
    assert manifest["target_limits_ns"] == limits
    assert manifest["split_separation_seconds"]["train"] == pytest.approx(
        (int(raw.timestamps[ranges["validation"][0]])
         - int(raw.timestamps[data.datasets["train"].target_indices.max()]))/1e9)
    # And it really is a different split from the walk-forward one, not the same thing renamed.
    assert manifest["ranges"] != walk_forward_split(raw, wf_config(grid_csv).data)[0]
