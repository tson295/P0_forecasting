import numpy as np
import pandas as pd

from p0.config import HMAX_SEC
from p0.data import asof_index, check_ohlcv, read_ohlcv_csv, to_b0_frame
from p0.split import Partition, check_fold, make_final, make_folds, utc_ts


def test_read_csv_skips_truncated_last_line(tmp_path, hf):
    p = tmp_path / "hf.csv"
    text = hf.to_csv(index=False)
    p.write_text(text[: len(text) - 17], encoding="utf-8")  # cắt cụt dòng cuối như file 2 MiB
    df = read_ohlcv_csv(p)
    assert len(df) in (len(hf) - 1, len(hf))
    rep = check_ohlcv(df)
    assert rep["ok"] and rep["duplicates"] == 0 and rep["gaps"] == 0 and rep["aligned"]


def test_adapter_uppercase(hf):
    b0 = to_b0_frame(hf)
    assert list(b0.columns) == ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    assert b0["timestamp"].dtype == np.int64


def test_partition_target_boundary(store):
    ts, el = store.ts, store.eligible
    end = utc_ts("2026-01-03")
    part = Partition(utc_ts("2026-01-02"), end)
    idx = part.origins(ts, el)
    assert ts[idx].max() + HMAX_SEC < end  # t + 3' < T_end
    assert ts[idx].max() == end - HMAX_SEC - 60  # origin cuối = T_end − 4'


def test_folds_disjoint_and_purge(store, folds):
    for f in folds:
        chk = check_fold(f, store.ts, store.eligible, 60)
        assert chk["ok"], chk
        assert f.val.start - f.es.end == 3600
        assert f.fit.end == f.es.start
        assert f.fit.start == store.first_origin_ts


def test_final_fold(store):
    final = make_final(store.first_origin_ts, "2026-01-04 00:00:00", store.last_ts + 60, 60)
    chk = check_fold(final, store.ts, store.eligible, 60)
    assert chk["ok"], chk
    idx = final.val.origins(store.ts, store.eligible)
    assert store.ts[idx].max() + HMAX_SEC <= store.last_ts


def test_asof_join_only_closed_bars(hf, lf):
    hf_ts = hf["timestamp"].to_numpy(np.int64)
    idx = asof_index(lf["timestamp"].to_numpy(np.int64), hf_ts)
    ok = idx >= 0
    assert (lf["timestamp"].to_numpy()[idx[ok]] <= hf_ts[ok]).all()
    # nhãn T = bar 1' cuối của nhóm (T−4..T]: LF open == HF open của bar T−4
    t = lf["timestamp"].iloc[3]
    assert lf["open"].iloc[3] == hf.set_index("timestamp").loc[t - 240, "open"]
