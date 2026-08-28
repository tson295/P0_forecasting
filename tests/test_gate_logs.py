"""Test cho các sửa sau checker review: gate --smoke/--allow-cpu chỉ với data tổng hợp, checksum §6.1 bắt buộc, config_hash không phụ
thuộc root, check_ohlcv phát hiện gap/dup, schema cố định của champion_log/log.csv, latency LSTM shared."""
from argparse import Namespace

import pandas as pd
import pytest

from p0 import cli
from p0.config import RunConfig
from p0.data import check_ohlcv, read_ohlcv_csv, verify_checksums, write_checksums
from p0.logs import CHAMPION_FIELDS, LOG_FIELDS, log_champion, log_run


def _cfg(tmp_path, label):
    return RunConfig(dataset_label=label, hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"], test_start="2026-01-04", root=str(tmp_path))


def test_gate_refuses_smoke_and_cpu_on_real_dataset(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "MEMORY.md").write_text("TRAINING: LOCKED\n", encoding="utf-8")
    real = _cfg(tmp_path, "btc_15d")
    with pytest.raises(SystemExit):  # --smoke không được bỏ khóa trên data thật
        cli.gate(real, Namespace(smoke=True, allow_cpu=False), ["lgbm"])
    with pytest.raises(SystemExit):  # --allow-cpu = CPU training trên data thật → cấm
        cli.gate(real, Namespace(smoke=False, allow_cpu=True), ["lgbm"])
    with pytest.raises(SystemExit):  # TRAINING: LOCKED
        cli.gate(real, Namespace(smoke=False, allow_cpu=False), ["lgbm"])
    with pytest.raises(SystemExit):
        cli.model_for(real, "lgbm", allow_cpu=True)
    cli.gate(_cfg(tmp_path, "synthetic_x"), Namespace(smoke=True, allow_cpu=True), ["lgbm"])  # data tổng hợp: đi qua


def test_config_hash_independent_of_root(tmp_path):
    a = _cfg(tmp_path / "a", "btc_15d")
    b = _cfg(tmp_path / "b", "btc_15d")
    assert a.hash() == b.hash()
    c = RunConfig(**{**a.to_dict(), "seeds": (1, 2, 3)})
    assert c.hash() != a.hash()


def test_check_ohlcv_detects_gap_and_duplicates(tmp_path, hf):
    gap = pd.concat([hf.iloc[:100], hf.iloc[104:]])  # bỏ 4 bar → một gap 300 s
    rep = check_ohlcv(gap)
    assert rep["gaps"] == 1 and rep["max_gap_sec"] == 300 and not rep["ok"]
    p = tmp_path / "dup.csv"
    pd.concat([hf.iloc[:200], hf.iloc[50:52]]).to_csv(p, index=False)
    rep = check_ohlcv(read_ohlcv_csv(p))  # drop_duplicates xảy ra khi đọc → vẫn phải đếm được
    assert rep["duplicates"] == 2 and not rep["ok"]


def test_checksums_relative_and_verified(tmp_path, hf):
    (tmp_path / "data").mkdir()
    p = tmp_path / "data" / "hf.csv"
    hf.to_csv(p, index=False)
    out = tmp_path / "data" / "data_checksums.json"
    payload = write_checksums("synthetic_t", {"hf": p}, {"hf": {}}, out, root=tmp_path)
    assert payload["files"]["hf"]["path"] == "data/hf.csv"  # tương đối root → verify được trên máy khác
    ok, problems = verify_checksums(out, tmp_path, label="synthetic_t")
    assert ok, problems
    ok, _ = verify_checksums(out, tmp_path, label="other_label")
    assert not ok
    p.write_text(p.read_text(encoding="utf-8") + "x", encoding="utf-8")
    ok, problems = verify_checksums(out, tmp_path, label="synthetic_t")
    assert not ok and "sha256" in problems[0]
    out.unlink()
    with pytest.raises(SystemExit):  # thiếu checksum → load_store từ chối (§6.1)
        cli.load_store(_cfg(tmp_path, "synthetic_t"), need_lf=False, verify=True)


def test_champion_and_run_log_fixed_schema(tmp_path):
    log_champion(tmp_path, {"model": "lgbm", "decision": "champion ban đầu", "champion_after": "lgbm"})
    log_champion(tmp_path, {"model": "xgb", "champion_before": "lgbm", "MedianGain_vs_champion": 0.01, "decision": "giữ",
                            "champion_after": "lgbm", "cột_thừa": 1})
    df = pd.read_csv(tmp_path / "champion_log.csv")
    assert list(df.columns) == CHAMPION_FIELDS and len(df) == 2
    log_run(tmp_path, {"exp_id": "a", "step": "x"})
    log_run(tmp_path, {"exp_id": "b", "MedianGain": 0.1, "gain_cells": "[]"})
    df = pd.read_csv(tmp_path / "log.csv")
    assert list(df.columns) == LOG_FIELDS and len(df) == 2 and df["timestamp"].notna().all()


def test_latency_lstm_shared(store, folds):
    from p0.harness import ColSet, run_config
    from p0.latency import measure_tabular
    from p0.models import make_model

    model = make_model("lstm", {"device": "cpu", "context": 8, "hidden": 4, "max_epochs": 1, "batch_size": 64}, allow_cpu=True)
    cs = ColSet(store.b0_names[:4], ("ret_60",))
    run = run_config(store, model, cs, folds[:1], rounds=(1, 1, 1), seed=1)
    lat = measure_tabular(run, warmup=2, max_origins=6, model=model)
    assert len(lat) == 3 and lat["shared"].all() and set(lat["h"]) == {1, 2, 3}
    assert lat["predict_device"].iloc[0] == "CPU" and lat["lib_version"].iloc[0].startswith("torch")
