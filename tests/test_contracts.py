"""Frozen contracts: capacity, HFformer window-local normalization, metrics, exports."""
import numpy as np
import pytest
import torch

from src.config import Config
from src.data.dataset import prepare_data
from src.data.preprocessing import RAW_COLUMNS
from src.models import build_model
from src.models.hfformer import WindowLocalZScore
from src.utils.metrics import HORIZON_LABELS, ReturnMetrics, metrics_from_arrays
from src.utils.predictions import PREDICTION_COLUMNS, predict, prediction_frame
from tests.test_data import write_csv

CPU = torch.device("cpu")
CAPACITY = dict(ofi_lstm=(20, 55491), hfformer=(38, 22026), patchtst=(40, 477059),
                moderntcn=(40, 50568195), lit=(40, 736547))
METRIC_KEYS = ("rmse", "mae", "r2", "rmse_e0", "rmse_gain_vs_e0")
PRICE_COLUMNS = [c for c in RAW_COLUMNS if "price" in c]
# Section Y spells these out; comparing the export to PREDICTION_COLUMNS alone would
# only prove the exporter agrees with itself.
SPEC_COLUMNS = ["origin_index", "origin_timestamp_ns", "origin_mid"] + [
    f"{field}_{h}" for h in ("1m", "2m", "3m")
    for field in ("target_index", "target_timestamp_ns", "target_mid",
                  "true_return", "pred_return", "pred_mid")]


def prepared(path, model):
    config = Config(model=model)
    config.data.csv_path = str(path)
    return prepare_data(config)


def corpus_shift(path, model):
    """prepare_data before/after validation+test prices are multiplied by ten."""
    frame = write_csv(path)
    before = prepared(path, model)
    train_hi = before.metadata["split_manifest"]["ranges"]["train"][1]
    frame.loc[train_hi:, PRICE_COLUMNS] *= 10
    frame.to_csv(path, index=False)
    return before, prepared(path, model)


def train_batch(data, count=16):
    return torch.stack([data.datasets["train"][i][0] for i in range(count)])


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    path = tmp_path_factory.mktemp("contracts")/"data.csv"
    write_csv(path)
    return prepared(path, "e0")


@pytest.mark.parametrize("name", CAPACITY)
def test_frozen_parameter_counts(name):
    channels, parameters = CAPACITY[name]
    model = build_model(name, 49, channels)
    assert sum(p.numel() for p in model.parameters()) == parameters


def test_hfformer_holds_no_corpus_statistics_and_is_affine_invariant():
    torch.manual_seed(0)
    model = build_model("hfformer", 49, 38).eval()
    # A saved mean/scale could only live in a buffer or parameter; HFformer has neither.
    assert list(model.buffers()) == []
    assert list(model.normalization.parameters()) == []
    x = torch.randn(4, 49, 38)*torch.rand(38).mul(5).add(.5)+torch.randn(38)*100
    scale, shift = torch.rand(38).mul(9).add(.5), torch.randn(38)*1000
    with torch.no_grad():
        rescaled, base = model(x*scale+shift), model(x)
    # Window-local z-score absorbs any per-feature affine rescale of the whole window;
    # eps=1e-5 leaves only a ~3e-6 residue against an O(0.5) output, while corpus
    # statistics would move the prediction by O(0.1).
    torch.testing.assert_close(rescaled, base, rtol=1e-4, atol=1e-4)


def test_window_local_zscore_matches_manual_unbiased_false_eps_reference():
    normalization = WindowLocalZScore()
    torch.manual_seed(1)
    x = torch.randn(3, 49, 7)*torch.rand(7).add(.5)+torch.randn(7)*50
    z = normalization(x)
    # Float64 oracle: numpy's std is already the population (ddof=0) statistic.
    array = x.double().numpy()
    reference = (array-array.mean(axis=1, keepdims=True))/(array.std(axis=1, keepdims=True)+1e-5)
    np.testing.assert_allclose(z.numpy(), reference, rtol=1e-4, atol=1e-4)
    assert z.mean(dim=1).abs().max() < 1e-5
    assert (z.std(dim=1, unbiased=False)-1).abs().max() < 1e-4
    # Sample (n-1) statistics would be 1% off at T=49, so this pins unbiased=False.
    assert (z.std(dim=1, unbiased=True)-1).abs().max() > 1e-2
    # A population std of exactly 1e-5 halves under eps=1e-5: pins the epsilon itself.
    column = x[:, :, :1]
    tiny = (column-column.mean(1, keepdim=True))/column.std(1, keepdim=True, unbiased=False)*1e-5
    assert normalization(tiny).std(dim=1, unbiased=False).sub(.5).abs().max() < 1e-3


def test_hfformer_samples_are_independent():
    torch.manual_seed(2)
    model = build_model("hfformer", 49, 38).eval()
    x = torch.randn(4, 49, 38)*1000+30000
    with torch.no_grad():
        torch.testing.assert_close(model(x[:1]), model(x)[:1], rtol=1e-5, atol=1e-6)


def test_hfformer_ignores_validation_and_test_corpus(tmp_path):
    before, after = corpus_shift(tmp_path/"hfformer.csv", "hfformer")
    assert before.metadata["preprocessing"]["standardizer"] is None
    assert after.metadata["preprocessing"]["standardizer"] is None
    x_before, x_after = train_batch(before), train_batch(after)
    torch.testing.assert_close(x_after, x_before, rtol=0, atol=0)
    torch.manual_seed(3)
    model = build_model("hfformer", before.history_rows, before.channels).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x_after), model(x_before), rtol=0, atol=0)


def test_ofi_lstm_keeps_a_train_only_standardizer(tmp_path):
    before, after = corpus_shift(tmp_path/"ofi_lstm.csv", "ofi_lstm")
    scaler = before.metadata["preprocessing"]["standardizer"]
    assert scaler is not None and scaler["fitted_on"] == "train_only"
    assert len(scaler["mean"]) == len(scaler["scale"]) == before.channels
    # Fitted on train rows only, so a ten-fold validation/test shift changes nothing.
    assert after.metadata["preprocessing"]["standardizer"] == scaler
    torch.testing.assert_close(train_batch(after), train_batch(before), rtol=0, atol=0)


def test_metric_contract_for_perfect_and_e0_predictions():
    y = np.random.default_rng(0).normal(scale=1e-4, size=(512, 3))
    perfect = metrics_from_arrays(y, y)
    zero = metrics_from_arrays(np.zeros_like(y), y)
    assert perfect["rmse"] == [0., 0., 0.]
    assert perfect["rmse_gain_vs_e0"] == [1., 1., 1.]
    assert zero["rmse"] == zero["rmse_e0"]
    assert zero["rmse_gain_vs_e0"] == [0., 0., 0.]
    np.testing.assert_allclose(zero["rmse_e0"], np.sqrt((y*y).mean(axis=0)), rtol=1e-12, atol=0)
    for result in (perfect, zero):
        assert set(METRIC_KEYS) <= set(result) and all(len(result[k]) == 3 for k in METRIC_KEYS)
        assert not any("r2_os" in key for key in result)  # Standard R2 only.
    target = torch.from_numpy(y)
    for prediction, gain in ((target, 1.), (torch.zeros_like(target), 0.)):
        streaming = ReturnMetrics(CPU)
        streaming.update(prediction, target)
        result = streaming.compute()
        assert result["rmse_gain_vs_e0"] == [gain]*3
        assert set(METRIC_KEYS) <= set(result) and not any("r2_os" in key for key in result)
        np.testing.assert_allclose(result["rmse_e0"], np.sqrt((y*y).mean(axis=0)), rtol=1e-12, atol=0)


def test_metric_formulas_match_an_independent_oracle():
    rng = np.random.default_rng(2)
    # Nonzero mean on purpose: standard R2 and R2_OS coincide only when mean(y) == 0.
    y = rng.normal(loc=3e-4, scale=1e-4, size=(600, 3))
    p = rng.normal(scale=1e-4, size=(600, 3))
    result = metrics_from_arrays(p, y)
    rmse, rmse_e0 = np.sqrt(((p-y)**2).mean(axis=0)), np.sqrt((y*y).mean(axis=0))
    for key, oracle in (("rmse", rmse), ("mae", np.abs(p-y).mean(axis=0)),
                        ("r2", 1-((p-y)**2).sum(axis=0)/((y-y.mean(axis=0))**2).sum(axis=0)),
                        ("rmse_e0", rmse_e0), ("rmse_gain_vs_e0", 1-rmse/rmse_e0)):
        np.testing.assert_allclose(result[key], oracle, rtol=1e-12, atol=0)
    # Checkpoint selection reads mean_mse, so it must be the mean over the 3 horizons.
    np.testing.assert_allclose(result["mean_mse"], ((p-y)**2).mean(), rtol=1e-12, atol=0)
    # R2_OS would divide by sum(y^2) and score the zero prediction at exactly 0.
    assert max(metrics_from_arrays(np.zeros_like(y), y)["r2"]) < -1
    streaming = ReturnMetrics(CPU)
    for lo, hi in ((0, 100), (100, 450), (450, 600)):  # unequal batches stay sample-weighted
        streaming.update(torch.from_numpy(p[lo:hi]), torch.from_numpy(y[lo:hi]))
    merged = streaming.compute()
    assert merged["samples"] == len(y)
    for key in METRIC_KEYS:
        np.testing.assert_allclose(merged[key], result[key], rtol=1e-10, atol=0)


def test_e0_predicts_zero_return_and_unchanged_mid(synthetic):
    dataset = synthetic.datasets["validation"]
    model = build_model("e0", synthetic.history_rows, synthetic.channels)
    predictions = predict(model, dataset, CPU)
    assert predictions.shape == (len(dataset), 3) and (predictions == 0).all()
    frame = prediction_frame(dataset, synthetic.raw, predictions)
    for h in HORIZON_LABELS:
        np.testing.assert_array_equal(frame[f"pred_return_{h}"].to_numpy(), 0.)
        np.testing.assert_array_equal(frame[f"pred_mid_{h}"].to_numpy(), frame["origin_mid"].to_numpy())


def test_prediction_export_schema_order_and_mid_reconstruction(synthetic):
    dataset = synthetic.datasets["test"]
    predictions = np.random.default_rng(1).normal(scale=1e-3, size=(len(dataset), 3))
    frame = prediction_frame(dataset, synthetic.raw, predictions)
    assert list(frame.columns) == SPEC_COLUMNS == PREDICTION_COLUMNS
    assert len(frame) == len(dataset)
    assert (np.diff(frame["origin_timestamp_ns"].to_numpy()) > 0).all()
    origin_mid = frame["origin_mid"].to_numpy()
    np.testing.assert_array_equal(frame["origin_timestamp_ns"].to_numpy(),
                                  synthetic.raw.timestamps[dataset.origins])
    for k, h in enumerate(HORIZON_LABELS):
        np.testing.assert_array_equal(frame[f"pred_return_{h}"].to_numpy(), predictions[:, k])
        np.testing.assert_array_equal(frame[f"pred_mid_{h}"].to_numpy(),
                                      origin_mid*np.exp(predictions[:, k]))
        np.testing.assert_array_equal(frame[f"true_return_{h}"].to_numpy(),
                                      np.log(frame[f"target_mid_{h}"].to_numpy()/origin_mid))
