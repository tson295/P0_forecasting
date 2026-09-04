"""Data 2 năm + split rolling_spread (quyết định user 2026-09-04) — khoá các yêu cầu §15:

(1) alias `ts` → `timestamp`; (2) ts/timestamp lệch → lỗi; (3) ingest chuỗi 60 s dài (không dup/gap); (4) dẫn xuất LF 5' tất định,
causal; (5) nhóm 5' thiếu bar ở đầu/cuối bị bỏ; (6) bar 5' nhãn T không bao giờ thấy trước T; (7)–(13) split rải đều: đúng 5 fold, VAL có thứ tự
và tách rời, không liền kề, FIT đúng 120 ngày, ES 5 ngày − purge, VAL 3 ngày, TEST 30 ngày cuối; (14) target không vượt biên partition;
(15) sentinel TEST không đổi; (16) fold-parallel tương thích; (17) config 15 ngày lịch sử không đổi.
"""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p0 import cli
from p0.config import RunConfig
from p0.data import asof_index, check_ohlcv, derive_lf_5min, file_sha256, read_ohlcv_csv, write_lf_csv
from p0.features_short import compute_short
from p0.split import DAY_SEC, RollingSpec, check_fold, make_folds, make_rolling_from_end, make_rolling_spread
from p0.synthetic import make_hf, make_lf

ROOT = Path(__file__).resolve().parents[1]
DAY = DAY_SEC


# ============================================================================= (1)(2)(3) ingestion
def _write(tmp_path, df, name="hf.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_ts_alias_is_renamed_in_memory_and_datetime_verified(tmp_path):
    hf = make_hf(n_days=0.5, seed=1)
    raw = hf.rename(columns={"timestamp": "ts"})
    raw["datetime"] = raw["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")  # như file 2 năm
    p = _write(tmp_path, raw)
    df = read_ohlcv_csv(p)
    assert "timestamp" in df.columns and "ts" not in df.columns and len(df) == len(hf)
    assert (df["timestamp"].to_numpy() == hf["timestamp"].to_numpy()).all()
    epoch = (df["datetime"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    assert str(df["datetime"].dt.tz) == "UTC" and (epoch == df["timestamp"]).all()
    # datetime lệch epoch → lỗi (không tin cột chữ)
    bad = raw.copy()
    bad.loc[5, "datetime"] = "2030-01-01 00:00:00+00:00"
    with pytest.raises(ValueError, match="datetime"):
        read_ohlcv_csv(_write(tmp_path, bad, "bad_dt.csv"))


def test_ts_and_timestamp_both_present_must_agree(tmp_path):
    hf = make_hf(n_days=0.2, seed=1)
    both = hf.copy()
    both.insert(0, "ts", both["timestamp"])
    df = read_ohlcv_csv(_write(tmp_path, both, "both.csv"))
    assert "ts" not in df.columns and len(df) == len(hf)
    both.loc[3, "ts"] = both.loc[3, "ts"] + 60
    with pytest.raises(ValueError, match="ts"):
        read_ohlcv_csv(_write(tmp_path, both, "disagree.csv"))


def test_exact_60s_long_series_ingests_cleanly(tmp_path):
    hf = make_hf(n_days=30.0, seed=2)  # 43.200 bar 60 s liên tục
    raw = hf.rename(columns={"timestamp": "ts"})
    df = read_ohlcv_csv(_write(tmp_path, raw, "long.csv"))
    rep = check_ohlcv(df)
    assert rep["ok"] and rep["rows"] == 43_200 and rep["gaps"] == 0 and rep["duplicates"] == 0 and rep["max_gap_sec"] == 60
    assert rep["amount_in_range"] and rep["amount_available"]


# ============================================================================= (4)(5)(6) LF 5' dẫn xuất
def test_derive_lf_is_deterministic_causal_and_matches_project_convention(tmp_path):
    hf = make_hf(n_days=2.0, seed=4)
    hf = hf.iloc[3:-2].reset_index(drop=True)  # đầu/cuối không khớp biên 5' → nhóm thiếu
    lf, meta = derive_lf_5min(hf)
    ref = make_lf(hf)  # quy ước cũ của project (nhãn T = bar cuối, T bội 300 s)
    assert len(lf) == len(ref) and (lf["timestamp"].to_numpy() == ref["timestamp"].to_numpy()).all()
    for c in ("open", "high", "low", "close", "volume", "amount"):
        assert np.allclose(lf[c].to_numpy(), ref[c].to_numpy())
    assert ((lf["timestamp"] % 300) == 0).all() and meta["dropped_incomplete_buckets"] >= 1 and meta["rows_lf"] == len(lf)
    # nhóm thiếu bar ở đầu/cuối bị bỏ, không bịa: bar 5' đầu tiên có nhãn ≥ ts đầu + 4', bar cuối ≤ ts cuối
    assert lf["timestamp"].iloc[0] >= hf["timestamp"].iloc[0] + 4 * 60 and lf["timestamp"].iloc[-1] <= hf["timestamp"].iloc[-1]
    # mỗi bar T = đúng 5 bar 1' (T−4' … T]: OHLC/volume/amount kiểm tay
    T = int(lf["timestamp"].iloc[7])
    win = hf[(hf["timestamp"] > T - 300) & (hf["timestamp"] <= T)]
    assert len(win) == 5 and np.isclose(lf["open"].iloc[7], win["open"].iloc[0]) and np.isclose(lf["close"].iloc[7], win["close"].iloc[-1])
    assert np.isclose(lf["high"].iloc[7], win["high"].max()) and np.isclose(lf["low"].iloc[7], win["low"].min())
    assert np.isclose(lf["volume"].iloc[7], win["volume"].sum()) and np.isclose(lf["amount"].iloc[7], win["amount"].sum())
    assert check_ohlcv(lf, step=300)["ok"]
    # tất định: dẫn xuất hai lần → cùng byte / cùng sha256
    s1 = write_lf_csv(lf, tmp_path / "a.csv")
    s2 = write_lf_csv(derive_lf_5min(hf)[0], tmp_path / "b.csv")
    assert s1 == s2 == file_sha256(tmp_path / "a.csv") and (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    # đọc lại đúng như LF cũ (alias/datetime)
    back = read_ohlcv_csv(tmp_path / "a.csv")
    assert (back["timestamp"].to_numpy() == lf["timestamp"].to_numpy()).all()


def test_5m_bar_never_visible_before_its_label(tmp_path):
    hf = make_hf(n_days=1.0, seed=5)
    lf, _ = derive_lf_5min(hf)
    ts_hf = hf["timestamp"].to_numpy(np.int64)
    j = asof_index(lf["timestamp"].to_numpy(np.int64), ts_hf)
    ok = j >= 0
    assert (lf["timestamp"].to_numpy()[j[ok]] <= ts_hf[ok]).all()  # T ≤ t luôn
    # origin t = T − 60 (bar 5' chưa đóng) phải thấy bar TRƯỚC đó, không phải T
    T = int(lf["timestamp"].iloc[10])
    t = int(np.flatnonzero(ts_hf == T - 60)[0])
    assert lf["timestamp"].iloc[j[t]] == T - 300 and lf["timestamp"].iloc[j[t + 1]] == T
    # feature 5' của C_short cũng chỉ đổi giá trị đúng tại bar T
    from p0.harness import Store

    store = Store(hf, lf)
    df = compute_short(store.grid, store.raw_lf, columns=("r5_2",))
    v = df["r5_2"].to_numpy()
    assert v[t] == v[t - 1] or (np.isnan(v[t]) and np.isnan(v[t - 1]))  # trước T: chưa có bar T
    assert v[t + 1] != v[t] or np.isnan(v[t])  # tại T: cập nhật


def test_cmd_derive_lf_writes_lf_and_sidecar_and_check_data_verifies_source(tmp_path):
    hf = make_hf(n_days=10.0, seed=6)  # đủ dài cho rolling_spread 2 fold (FIT 2 + ES 1 + VAL 1 + TEST 1, VAL tách rời)
    (tmp_path / "data").mkdir()
    hf.rename(columns={"timestamp": "ts"}).to_csv(tmp_path / "data" / "hf2y.csv", index=False)
    cfg = RunConfig(dataset_label="synthetic_2y", hf_csv="data/hf2y.csv", lf_csv="data/lf5m.csv", checksums="data/ck.json",
                    split={"mode": "rolling_spread", "n_folds": 2, "val_days": 1, "fit_days": 2, "es_days": 1, "test_days": 1, "purge_minutes": 60},
                    root=str(tmp_path), require_gpu=False)
    cli.cmd_derive_lf(cfg, Namespace(force=False))
    side = json.loads((tmp_path / "data" / "lf5m.derivation.json").read_text(encoding="utf-8"))
    assert side["source_sha256"] == file_sha256(tmp_path / "data" / "hf2y.csv") and side["lf_sha256"] == file_sha256(tmp_path / "data" / "lf5m.csv")
    with pytest.raises(SystemExit):  # đã có LF → không ghi đè khi không --force
        cli.cmd_derive_lf(cfg, Namespace(force=False))
    cli.cmd_check_data(cfg, Namespace(write_checksums=True))
    ck = json.loads((tmp_path / "data" / "ck.json").read_text(encoding="utf-8"))
    assert set(ck["files"]) == {"hf", "lf"} and ck["dataset_label"] == "synthetic_2y"
    store, folds, final, _ = cli.load_store(cfg)
    assert len(folds) == 2 and store.raw_lf is not None
    # HF khác (sidecar không khớp) → check-data hard fail
    hf.iloc[:-5].rename(columns={"timestamp": "ts"}).to_csv(tmp_path / "data" / "hf2y.csv", index=False)
    with pytest.raises(SystemExit):
        cli.cmd_check_data(cfg, Namespace(write_checksums=False))


# ============================================================================= (7)–(14) split rolling_spread
SPEC = RollingSpec(n_folds=5, val_days=3, fit_days=120, es_days=5, test_days=30, purge_minutes=60, mode="rolling_spread")


def _two_year():
    first = int(pd.Timestamp("2024-09-04 03:00:00", tz="UTC").timestamp())  # origin eligible đầu (sau warmup B0)
    last = int(pd.Timestamp("2026-09-03 16:29:00", tz="UTC").timestamp())
    return first, last


def test_spread_split_five_folds_ordered_separated_and_sized():
    first, last = _two_year()
    folds, final = make_rolling_spread(first, last, SPEC)
    assert len(folds) == 5
    t_end = last + 60
    assert final.val.end == t_end and final.val.start == t_end - 30 * DAY  # (13) TEST 30 ngày cuối
    starts = [f.val.start for f in folds]
    assert starts == sorted(starts) and all(b - a > 3 * DAY for a, b in zip(starts, starts[1:]))  # (8) thứ tự, không chồng
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) > 90 * DAY  # (9) rải đều, không liền kề: ~142 ngày giữa các VAL trên data 2 năm
    assert abs(max(gaps) - min(gaps)) <= 120  # xấp xỉ đều (làm tròn lưới phút)
    assert starts[-1] == final.val.start - 3 * DAY  # VAL muộn nhất kết thúc ngay trước TEST
    assert folds[0].fit.start >= first  # VAL sớm nhất còn đủ FIT + ES
    for f in folds:
        assert f.fit.end - f.fit.start == 120 * DAY  # (10) FIT đúng 120 ngày
        assert f.es.start == f.fit.end and f.es.end - f.es.start == 5 * DAY - 3600  # (11) ES 5 ngày − purge
        assert f.val.end - f.val.start == 3 * DAY and f.val.start - f.es.end == 3600  # (12) VAL 3 ngày, purge 60'
        assert f.fit.start % 60 == 0 and f.val.start % 60 == 0
    assert final.fit.end - final.fit.start == 120 * DAY and final.es.end == final.val.start - 3600  # final refit = 120 + 5 trước TEST
    # xấp xỉ mốc mong đợi cho data thật (không hard-code trong code)
    assert pd.Timestamp(folds[0].val.start, unit="s", tz="UTC").strftime("%Y-%m-%d") in ("2025-01-06", "2025-01-07")
    assert pd.Timestamp(folds[-1].val.start, unit="s", tz="UTC").strftime("%Y-%m-%d") == "2026-08-01"
    assert pd.Timestamp(final.val.start, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M") == "2026-08-04 16:30"


def test_spread_split_rejects_too_short_data_and_keeps_from_end_mode():
    first = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    with pytest.raises(ValueError):
        make_rolling_spread(first, first + 100 * DAY, SPEC)
    spec_fe = RollingSpec(n_folds=5, val_days=3, fit_days=40, es_days=5, test_days=30)
    folds, final = make_rolling_from_end(first, first + 100 * DAY, spec_fe)  # mode cũ vẫn chạy y nguyên
    assert len(folds) == 5 and all(b.val.start - a.val.start == 3 * DAY for a, b in zip(folds, folds[1:]))
    assert RollingSpec.from_dict({"mode": "rolling_spread", "fit_days": 120}).mode == "rolling_spread"
    with pytest.raises(ValueError):
        RollingSpec.from_dict({"mode": "expanding"})


def test_spread_split_on_synthetic_store_no_target_crosses_boundary():
    hf = make_hf(n_days=12.0, seed=7)
    lf = make_lf(hf)
    from p0.harness import Store

    store = Store(hf, lf)
    spec = RollingSpec(n_folds=3, val_days=1, fit_days=3, es_days=1, test_days=2, purge_minutes=60, mode="rolling_spread")
    folds, final = make_rolling_spread(store.first_origin_ts, store.last_ts, spec)
    assert len(folds) == 3
    for f in folds + [final]:
        chk = check_fold(f, store.ts, store.eligible, 60)
        assert chk["ok"], chk
        for part in (f.fit, f.es, f.val):
            idx = part.origins(store.ts, store.eligible)
            assert len(idx) and (store.ts[idx] + 180 < part.end).all() and (store.ts[idx] >= part.start).all()  # (14)
    vals = [f.val.start for f in folds]
    assert vals == sorted(vals) and min(b - a for a, b in zip(vals, vals[1:])) > 1 * DAY


# ============================================================================= (15)(16)(17) không đổi
def test_sentinel_and_fold_parallel_unchanged_with_spread_config(tmp_path, monkeypatch):
    hf = make_hf(n_days=12.0, seed=8)
    (tmp_path / "data").mkdir()
    hf.to_csv(tmp_path / "data" / "hf.csv", index=False)
    make_lf(hf).to_csv(tmp_path / "data" / "lf.csv", index=False)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "MEMORY.md").write_text("TRAINING: UNLOCKED\n", encoding="utf-8")
    cfg = RunConfig(dataset_label="synthetic_spread", hf_csv="data/hf.csv", lf_csv="data/lf.csv", checksums="data/ck.json",
                    split={"mode": "rolling_spread", "n_folds": 2, "val_days": 1, "fit_days": 3, "es_days": 1, "test_days": 2, "purge_minutes": 60},
                    root=str(tmp_path), require_gpu=False, models={"lgbm": {"n_jobs": 1, "n_estimators": 8, "min_child_samples": 20}})
    cli.cmd_check_data(cfg, Namespace(write_checksums=True))
    store, folds, final, _ = cli.load_store(cfg)
    assert len(folds) == 2 and final.val.end - final.val.start == 2 * DAY
    from p0 import fold_parallel
    from p0.harness import ColSet, run_config

    model = cli.model_for(cfg, "lgbm", allow_cpu=True)
    cs = ColSet(store.b0_names[:10])
    seq = run_config(store, model, cs, folds, rounds=(4, 4, 4), seed=1, keep_states=False)
    monkeypatch.setenv("P0_FOLD_WORKERS", "2")
    try:
        assert fold_parallel.configure(cfg, model, "lgbm", True) == 2
        par = run_config(store, model, cs, folds, rounds=(4, 4, 4), seed=1, keep_states=False)
        assert par.fold_names == seq.fold_names and np.allclose(par.rmse, seq.rmse, atol=1e-6)  # (16)
    finally:
        fold_parallel.shutdown()
        monkeypatch.delenv("P0_FOLD_WORKERS", raising=False)
        fold_parallel.configure(cfg, model, "lgbm", True)
    # (15) sentinel: final lần hai bị từ chối
    (cfg.exp_dir / "final").mkdir(parents=True, exist_ok=True)
    (cfg.exp_dir / "final" / "TEST_SENTINEL.json").write_text(json.dumps({"status": "completed", "started_at": "t0", "config_hash": cfg.hash()}), encoding="utf-8")
    with pytest.raises(SystemExit, match="TEST_ALREADY_RUN"):
        cli.cmd_final(cfg, Namespace(smoke=True, allow_cpu=True, latency_origins=None, force_test_rerun=False))


def test_historical_15d_config_unchanged():
    cfg = RunConfig.load(ROOT / "configs" / "p0_15d.json")
    assert cfg.split is None and cfg.val_days == ["2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31"]
    assert cfg.hf_csv == "data/BTC_hf_1min.csv" and cfg.checksums == "data/data_checksums.json" and cfg.experiments_dir == "experiments"
    first = int(pd.Timestamp("2026-01-19 02:46:00", tz="UTC").timestamp())
    folds = make_folds(first, cfg.val_days, cfg.purge_minutes, cfg.es_hours)
    assert len(folds) == 5 and folds[0].fit.start == first and folds[4].val.end - folds[4].val.start == DAY  # expanding, VAL 1 ngày
    full = RunConfig.load(ROOT / "configs" / "p0_full.json")
    assert full.hf_csv == "data/BTC_1m_2y.csv" and full.lf_csv == "data/BTC_5m_2y.csv" and full.checksums == "data/data_checksums_2y.json"
    assert full.split["mode"] == "rolling_spread" and full.split["fit_days"] == 120 and full.split["n_folds"] == 5 and full.split["test_days"] == 30
    assert full.dataset_label == "btc_1min_2y_2024-09-03_2026-09-03" and full.prev_run_dir == "experiments/15d"
