"""Nhánh run/ml-lstm-expanded — 2 thay đổi:

(1) Precompute feature ext của S0_m + Candidate_m một lần trước khi candidate search/training bắt đầu
    (`Store.ensure_ext` gọi MỘT LẦN, batched — thay vì add-one loop kích hoạt lại từng cột một):
    `s0.union_ext_columns` (hợp cột cần precompute) + `cmd_loop` (gọi trước calibrate/add-one khi scheduler tắt).
(2) Config ML+LSTM-only (`configs/p0_ml_lstm.json`): giống hệt `configs/p0_full.json` trừ phạm vi model,
    thứ tự champion replay cố định lgbm → xgb → cat → xgbrf → lstm.

Không đổi S0/Candidate/PI/selection/hyperparameter — chỉ tối ưu thực thi (khi nào feature được tính).
"""
import json
from argparse import Namespace
from pathlib import Path

import pytest

from p0 import cli
from p0.config import RunConfig
from p0.features_short import SHORT_COLUMNS
from p0.harness import ColSet
from p0.s0 import collision_audit, save_lock, union_ext_columns

ROOT = Path(__file__).resolve().parents[1]


def _counting_compute_short(monkeypatch):
    """Bọc `compute_short` để đếm số lần gọi + cột mỗi lần — không đổi giá trị trả về."""
    import p0.features_short as fs

    orig = fs.compute_short
    calls: list[tuple] = []

    def wrapper(g, lf=None, columns=None):
        calls.append(tuple(columns) if columns is not None else tuple(fs.SHORT_COLUMNS))
        return orig(g, lf, columns=columns)

    monkeypatch.setattr(fs, "compute_short", wrapper)
    return calls


# ============================================================================= (1a) Store.ensure_ext batches
def test_ensure_ext_batches_missing_short_columns_in_one_compute_short_call(store, monkeypatch):
    calls = _counting_compute_short(monkeypatch)
    cols = list(SHORT_COLUMNS[:6])
    for c in cols:
        store._ext.pop(c, None)
    store.ensure_ext(cols)
    assert len(calls) == 1 and set(calls[0]) == set(cols)  # MỘT lời gọi, mọi cột gộp chung — không tính từng cột một
    calls.clear()
    store.ensure_ext(cols)  # đã cache hết → gọi lại không tính thêm
    assert calls == []


# ============================================================================= (1b) union cột cần precompute
def test_union_ext_columns_merges_across_models_and_skips_missing_or_non_precompute(tmp_path, store):
    s0 = {
        "lgbm": ColSet(store.b0_names[:3], (SHORT_COLUMNS[0],), (SHORT_COLUMNS[0],)),  # SHORT_COLUMNS[0] đã khoá trong S0_lgbm
        "xgb": ColSet(store.b0_names[:3]),  # không ext khoá → Candidate_xgb = trọn 5 cột
        "tfm": ColSet((), ()),  # có lock-s0 nhưng KHÔNG thuộc PRECOMPUTE_MODELS → phải bị bỏ qua
    }
    rep = collision_audit(store, s0, SHORT_COLUMNS[:5], dataset_label="synthetic")
    save_lock(tmp_path, s0, rep)
    cols = union_ext_columns(tmp_path, ["lgbm", "xgb", "tfm", "cat"], "synthetic")  # "cat": chưa lock-s0 → bỏ qua
    assert set(cols) == set(SHORT_COLUMNS[:5])
    assert len(cols) == len(set(cols))  # không trùng
    assert union_ext_columns(tmp_path, ["tfm"], "synthetic") == ()  # tfm/autots không đổi ở nhánh này
    assert union_ext_columns(tmp_path, ["lgbm"], "other_dataset") == ()  # audit khác dataset → bỏ qua lặng lẽ (không hard_fail)


# ============================================================================= (1c) cmd_loop: precompute trước add-one thật
def _small_lgbm_cfg(tmp_path) -> RunConfig:
    from p0.synthetic import make_hf, make_lf

    hf = make_hf(n_days=6.0, seed=7)
    lf = make_lf(hf)
    (tmp_path / "data").mkdir()
    hf.to_csv(tmp_path / "data" / "hf.csv", index=False)
    lf.to_csv(tmp_path / "data" / "lf.csv", index=False)
    cfg = RunConfig(dataset_label="synthetic_precompute", hf_csv="data/hf.csv", lf_csv="data/lf.csv",
                    val_days=["2026-01-03", "2026-01-04"], test_start="2026-01-05", root=str(tmp_path), require_gpu=False,
                    short_candidates=list(SHORT_COLUMNS[:4]), model_order=["lgbm"],
                    models={"lgbm": {"n_jobs": 1, "n_estimators": 8, "min_child_samples": 5}})
    cli.cmd_check_data(cfg, Namespace(write_checksums=True))
    ColSet(tuple(cli.load_store(cfg)[0].b0_names[:5])).save(cfg.exp_dir / "b0_star.json")
    cli.cmd_lock_s0(cfg, Namespace(data_config=None, max_rows=2000))
    return cfg


def test_cmd_loop_precomputes_all_candidate_ext_columns_once_before_search(tmp_path, monkeypatch):
    cfg = _small_lgbm_cfg(tmp_path)
    calls = _counting_compute_short(monkeypatch)
    args = Namespace(model="lgbm", smoke=True, allow_cpu=True, max_candidates=None, no_standalone=True,
                     latency_origins=None, resume=False)
    cli.cmd_loop(cfg, args)
    # calibrate → seed_noise → add-one (4 candidate) → prune PI → confirmation (raw/pruned) đều CHỌN cột đã precompute;
    # nếu bỏ bước precompute, add-one loop sẽ kích hoạt compute_short RIÊNG cho từng candidate mới (4 lời gọi thay vì 1).
    assert len(calls) == 1, f"compute_short phải chỉ được gọi MỘT LẦN (precompute), thấy {len(calls)} lần: {calls}"
    assert set(calls[0]) == set(SHORT_COLUMNS[:4])
    assert (cfg.exp_dir / "wins" / "lgbm.json").exists()  # loop vẫn chạy xong bình thường, kết quả không đổi


# ============================================================================= (2) config ML+LSTM-only
FULL_CFG = ROOT / "configs" / "p0_full.json"
ML_LSTM_CFG = ROOT / "configs" / "p0_ml_lstm.json"


@pytest.mark.skipif(not (FULL_CFG.exists() and ML_LSTM_CFG.exists()), reason="thiếu file config")
def test_ml_lstm_config_identical_to_full_except_model_scope_and_experiments_dir():
    full = RunConfig.load(FULL_CFG)
    ml = RunConfig.load(ML_LSTM_CFG)
    assert ml.model_order == ["lgbm", "xgb", "cat", "xgbrf", "lstm"]
    assert set(ml.models) == {"lgbm", "xgb", "cat", "xgbrf", "lstm"}
    for m in ml.models:
        assert ml.models[m] == full.models[m]  # hyperparameter model không đổi
    same_fields = ("dataset_label", "hf_csv", "lf_csv", "checksums", "split", "purge_minutes", "calib_seed",
                   "eval_seeds", "selection_seed", "eps_floor_pp", "prev_run_dir", "fold_workers", "gpu_devices",
                   "gpu_slots_per_device", "max_branches", "defer_champion", "short_candidates", "require_gpu")
    for f in same_fields:
        assert getattr(ml, f) == getattr(full, f), f  # data/split/seed/eps/GPU giống hệt config full
    assert ml.experiments_dir != full.experiments_dir  # experiments riêng — không lẫn với run 8 model


@pytest.mark.skipif(not ML_LSTM_CFG.exists(), reason="thiếu file config")
def test_ml_lstm_champion_replay_order_is_fixed_lgbm_xgb_cat_xgbrf_lstm():
    from p0.orchestrate import LOOP_MODELS, build_dag

    cfg = RunConfig.load(ML_LSTM_CFG)
    order = ["lgbm", "xgb", "cat", "xgbrf", "lstm"]
    assert cli.representatives_expected(cfg) == order  # §3 thứ tự so champion — KHÔNG theo thứ tự chạy xong
    assert all(m in LOOP_MODELS for m in cfg.model_order)
    branches = build_dag(cfg.model_order)
    # chỉ 5 nhánh loop — KHÔNG tfm-final/autots-search (không có tfm/autots_wr/autots_mr trong models)
    assert [b.name for b in branches] == [f"loop:{m}" for m in order]
    assert all(b.kind == "loop" and not b.deps for b in branches)
