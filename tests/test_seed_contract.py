"""Hợp đồng seed giữa confirmation → artifact → ensemble (fix 2026-09-01).

1. Model tất định (TimesFM zero-shot): confirmation chỉ chạy 1 seed → `wins/*.json` phải ghi ĐÚNG seed đã chạy
   và chỉ có `*_seed0.npz`; `tfm-final` copy đúng file của nhánh thắng; `ensemble` không đi tìm seed1/seed2.
2. AutoTS-final: chọn candidate ở `selection_seed`; chỉ SAU KHI freeze mới chạy 3 `eval_seeds` để lấy RMSE̅/ε.
3. ε của AutoTS-final tính từ chính 3 bảng RMSE của nó.
4. `tfm_b0` / `tfm_ext` là alias TimesFM → `--allow-cpu` (synthetic) phải ép `device="cpu"`.
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
    for name in ("tfm", "tfm_b0", "tfm_ext"):
        m = cli.model_for(cfg, name, allow_cpu=True)
        assert m.device == "cpu" and m.train_device == "CPU", name
    for name in ("xgb", "xgbrf", "lstm"):
        assert cli.model_for(cfg, name, allow_cpu=True).train_device == "CPU", name
    assert cli.model_for(cfg, "lgbm", allow_cpu=True).config.device_type == "cpu"


# ----------------------------------------------------------------------------- (1) TimesFM tất định → ensemble
def test_deterministic_confirmation_runs_one_seed(store, folds):
    m = TimesFMModel(device="cpu", allow_cpu=True, context=512, batch_size=64, covariate_scope="ext", model=StubTFM())
    conf = confirm(store, m, ColSet((), ("ret_60",)), folds[:1], (5, 6, 7))
    assert [int(r.seed) for r in conf.runs] == [5]  # tất định → 1 run thật, không giả thêm seed
    assert np.allclose(conf.rmse_mean, conf.runs[0].rmse)


def _run_tfm_branch(cfg, store, folds, branch, monkeypatch, tmp_path):
    """Chạy `cmd_loop` thật cho một nhánh TimesFM (stub package) → sinh wins/<branch>.json + npz."""
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_store", lambda c, **k: (store, folds, None, None))
    monkeypatch.setattr(cli, "model_for", lambda c, name, allow_cpu: TimesFMModel(
        device="cpu", allow_cpu=True, context=512, batch_size=64,
        covariate_scope="b0star" if name == "tfm_b0" else "ext", name=name, model=StubTFM()))
    if branch == "tfm_b0":
        ColSet(store.b0_names[:4]).save(cfg.exp_dir / "b0_star.json")
    cli.cmd_loop(cfg, Namespace(model=branch, smoke=True, allow_cpu=True, max_candidates=1,
                                no_standalone=True, latency_origins=20))


def test_timesfm_branches_to_ensemble_without_seed_file_crash(tmp_path, store, folds, monkeypatch):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    _champion(exp, folds[:1])
    use_folds = folds[:1]
    for branch in ("tfm_ext", "tfm_b0"):
        _run_tfm_branch(cfg, store, use_folds, branch, monkeypatch, tmp_path)
        w = json.loads((exp / "wins" / f"{branch}.json").read_text(encoding="utf-8"))
        # eval_seeds ghi ĐÚNG số run thật (tất định → 1), khớp số file npz
        assert w["eval_seeds"] == [cfg.eval_seeds[0]], (branch, w["eval_seeds"])
        assert (exp / "wins" / f"{branch}_seed0.npz").exists()
        assert not (exp / "wins" / f"{branch}_seed1.npz").exists()
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    fin = json.loads((exp / "wins" / "tfm.json").read_text(encoding="utf-8"))
    assert fin["eval_seeds"] == [cfg.eval_seeds[0]] and (exp / "wins" / "tfm_seed0.npz").exists()

    # ensemble: thành viên thứ hai (lgbm) dùng CHÍNH origin của tfm để hai bên khớp origin
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
