import pytest
from src.config import Config
import train


def test_smoke_cannot_reach_fit_or_probe(monkeypatch, tmp_path):
    from tests.test_data import write_csv
    csv = tmp_path/"synthetic.csv"; write_csv(csv)
    def forbidden(*args, **kwargs):
        raise AssertionError("Training/probing invoked during smoke")
    monkeypatch.setattr(train, "fit", forbidden)
    import src.training.trainer as trainer
    monkeypatch.setattr(trainer, "probe_batch_size", forbidden)
    train.main(["--csv", str(csv), "--model", "ofi_lstm", "--smoke-test", "--backward",
                "--auto-batch-size", "--output", str(tmp_path/"smoke")])
    assert (tmp_path/"smoke/smoke_ofi_lstm.json").exists()


def test_config_unknown_options_and_invalid_flags():
    config = Config(); config.data.history_seconds = 0
    with pytest.raises(ValueError, match="positive number of seconds"): config.validate()
    config = Config(); config.data.duplicate_timestamp_policy = "drop_all"
    with pytest.raises(ValueError, match="duplicate_timestamp_policy"): config.validate()
    args = train.parser().parse_args(["--backward"])
    with pytest.raises(ValueError, match="only valid"):
        train.resolve_config(args)
