"""Hợp đồng seed giữa confirmation → artifact → ensemble (fix 2026-09-01).

1. Model tất định (zero-shot, tham chiếu): confirmation chỉ chạy 1 seed → `wins/*.json` ghi ĐÚNG seed đã chạy
   và chỉ có `*_seed0.npz`; `ensemble` không đi tìm seed1/seed2.
2. AutoTS-final: chọn candidate ở `selection_seed`; chỉ SAU KHI freeze mới chạy 3 `eval_seeds` để lấy RMSE̅/ε.
3. ε của AutoTS-final tính từ chính 3 bảng RMSE của nó.
4. `tfm` (LoRA) với `--allow-cpu` (synthetic) phải ép `device="cpu"` và trỏ adapter_dir vào experiments/<run>/lora.
"""
import json
from argparse import Namespace

import numpy as np
import pandas as pd

from p0 import cli
from p0.config import RunConfig
from p0.harness import ColSet
from p0.loop import confirm
from p0.models_tfm import TimesFMModel

from test_tfm_autots import StubTFM


def _cfg(tmp_path, **kw):
    kw.setdefault("eval_seeds", (5, 6, 7))
    kw.setdefault("selection_seed", 5)
    return RunConfig(dataset_label="synthetic_seed", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"],
                     test_start="2026-01-04", root=str(tmp_path), models={"tfm": {"device": "cpu"}}, **kw)


def _champion(exp, folds):
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "champion.json").write_text(json.dumps(
        {"model": "lgbm", "colset": {"b0": [], "ext": []}, "rmse_mean": [[95.0, 135.0, 165.0]] * len(folds),
         "eps": 0.02, "e0": [[100.0, 140.0, 170.0]] * len(folds)}), encoding="utf-8")


# ----------------------------------------------------------------------------- (4) alias CPU
def test_tfm_aliases_forced_to_cpu_on_synthetic(tmp_path):
    cfg = _cfg(tmp_path)
    m = cli.model_for(cfg, "tfm", allow_cpu=True)
    assert m.device == "cpu" and m.train_device == "CPU" and m.name == "tfm"
    assert str(m.adapter_dir).endswith("lora")  # adapter LoRA là artifact trong experiments/<run>/lora
    for name in ("xgb", "xgbrf", "lstm"):
        assert cli.model_for(cfg, name, allow_cpu=True).train_device == "CPU", name
    assert cli.model_for(cfg, "lgbm", allow_cpu=True).config.device_type == "cpu"


# ----------------------------------------------------------------------------- (1) TimesFM tất định → ensemble
def test_deterministic_confirmation_runs_one_seed(store, folds):
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope="ext", model=StubTFM())
    conf = confirm(store, m, ColSet((), ("ret_60",)), folds[:1], (5, 6, 7))
    assert [int(r.seed) for r in conf.runs] == [5]  # tất định → 1 run thật, không giả thêm seed
    assert np.allclose(conf.rmse_mean, conf.runs[0].rmse)


def test_deterministic_member_to_ensemble_without_seed_file_crash(tmp_path, store, folds, monkeypatch):
    """Thành viên tất định (1 seed, chỉ *_seed0.npz) + thành viên 2 seed → ensemble lấy min số seed, không đi tìm seed1/seed2."""
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    use_folds = folds[:1]
    _champion(exp, use_folds)
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope="ext", model=StubTFM())
    conf = confirm(store, m, ColSet((), ("ret_60",)), use_folds, cfg.eval_seeds)
    w = cli._save_win(exp, "tfm", conf, 0.005, "prune", use_folds)
    assert w["eval_seeds"] == [cfg.eval_seeds[0]] and (exp / "wins" / "tfm_seed0.npz").exists()
    assert not (exp / "wins" / "tfm_seed1.npz").exists()
    z = np.load(exp / "wins" / "tfm_seed0.npz")
    n_fold = len([k for k in z.files if k.startswith("idx_")])
    kw = {}
    for i in range(n_fold):
        kw[f"idx_{i}"] = z[f"idx_{i}"]
        kw[f"yhat_{i}"] = z[f"yhat_{i}"] * 0.5
    for k in range(2):  # lgbm stochastic: 2 seed → ensemble lấy min số seed
        np.savez_compressed(exp / "wins" / f"lgbm_seed{k}.npz", **kw)
    (exp / "wins" / "lgbm.json").write_text(json.dumps(
        {"model": "lgbm", "colset": {"b0": [], "ext": []}, "rmse_mean": [[95.0, 135.0, 165.0]] * len(use_folds),
         "e0": [[100.0, 140.0, 170.0]] * len(use_folds), "eps": 0.02, "eval_seeds": [5, 6],
         "median_gain_vs_e0": 0.5}), encoding="utf-8")
    (exp / "wins" / "tfm.json").write_text(json.dumps({**w, "median_gain_vs_e0": 0.2}), encoding="utf-8")
    monkeypatch.setattr(cli, "load_store", lambda c, **k: (store, use_folds, None, None))
    cli.cmd_ensemble(cfg, Namespace(smoke=True, allow_cpu=True))  # KHÔNG được crash vì thiếu tfm_seed1.npz
    ens = json.loads((exp / "ensemble.json").read_text(encoding="utf-8"))
    assert "tfm" in ens["members"] and "lgbm" in ens["members"]
    assert np.isfinite(np.asarray(ens["rmse_mean"])).all()


# ----------------------------------------------------------------------------- (2)+(3) seed protocol AutoTS-final
def test_autots_selection_uses_selection_seed_then_confirms_on_eval_seeds(tmp_path, store, folds, monkeypatch):
    from test_autots_search import _prepare

    cfg, exp, seen = _prepare(tmp_path, store, folds, monkeypatch, ["ret_60"], ["bb_pctb_20"])
    real_run_config = cli.run_config
    calls = []

    def spy(store_, model, colset, folds_, rounds=None, seed=8586, keep_states=True):
        calls.append({"seed": int(seed), "ext": tuple(colset.ext), "keep": keep_states})
        return real_run_config(store_, model, colset, folds_, rounds=rounds, seed=seed, keep_states=keep_states)

    monkeypatch.setattr(cli, "run_config", spy)
    cli.cmd_autots_search(cfg, Namespace(smoke=True, allow_cpu=True))

    n_folds, n_cand = len(folds), 2 * 2  # 2 frozen set × 2 nhóm shift
    # pha CHỌN dùng keep_states=False, pha CONFIRMATION dùng keep_states=True (cần prediction cho ensemble)
    sel = [c for c in calls if not c["keep"]]
    conf = [c for c in calls if c["keep"]]
    assert len(sel) == n_cand * n_folds and {c["seed"] for c in sel} == {cfg.sel_seed}  # chọn CHỈ ở selection_seed
    assert len(conf) == len(cfg.eval_seeds) * n_folds and {c["seed"] for c in conf} == set(cfg.eval_seeds)
    win = json.loads((exp / "wins" / "autots.json").read_text(encoding="utf-8"))
    assert {c["ext"] for c in conf} == {tuple(win["colset"]["ext"])}  # chỉ winner đã freeze mới được confirm
    assert len(win["seed_rmse"]) == len(cfg.eval_seeds)  # đủ 3 bảng RMSE của winner
    # ε lấy từ chính AutoTS-final, KHÔNG mượn ε của probe (probe eps = 0.02 trong fixture)
    from p0.metrics import seed_noise_eps

    assert np.isclose(win["eps"], seed_noise_eps([np.array(t) for t in win["seed_rmse"]], cfg.eps_floor_pp))
    ch = pd.read_csv(exp / "champion_log.csv")
    assert (ch["model"] == "autots").any()  # AutoTS-final đi so champion bằng ε của chính nó
    df = pd.read_csv(exp / "autots_search.csv")
    assert "MedianGain_vs_E0_sel" in df.columns and bool(df["is_final"].fillna(False).any())
