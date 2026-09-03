"""TimesFM-LoRA (quyết định user 2026-09-03) — kiểm bằng stub thay package thật (timesfm chưa cài local):

11. LoRA: FIT chỉ để học, ES chỉ để chọn epoch, VAL không bao giờ vào training, adapter freeze trước candidate search;
12. thêm candidate KHÔNG train lại / KHÔNG đổi trọng số LoRA;   13. XReg search dùng CÙNG adapter đã freeze;
14. tfm-final: TFM-LoRA + XReg(win) vs TFM-LoRA native theo luật project (> +ε) — không phải "XReg vs TFM";
+ adapter tải lại từ đĩa tất định, predictor của từng fold nạp đúng adapter của fold đó, `loop --model tfm` end-to-end với stub.
Toán LoRA (`p0.lora`) và `train_forward` trên module TimesFM thật đã được canary local kiểm (audit_timesfm_lora.md §11).
"""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from p0 import cli
from p0 import models_tfm
from p0.config import RunConfig
from p0.features_short import SHORT_BY_NAME, SHORT_COLUMNS
from p0.harness import ColSet, run_config
from p0.loop import add_one_loop, confirm, prune_pi
from p0.lora import lora_state_dict, state_sha256
from p0.models_tfm import TimesFMLoRAModel


class StubLoRATFM:
    """Thay TimesFM thật: `.model` = nn.Module nhỏ có Linear `proj` (đích LoRA); mọi đường (train_forward, forecast,
    forecast_with_covariates) đi qua CHÍNH module đó → LoRA inject/adapter nạp có hiệu lực ở cả ba đường."""

    def __init__(self, L: int = 8):
        import torch

        torch.manual_seed(0)
        m = torch.nn.Module()
        m.proj = torch.nn.Linear(L, 3, bias=False)
        self.model, self.L, self.calls = m, L, []

    def train_forward(self, x):
        return self.model.proj(x[:, -self.L:] * 100.0)

    def _pred(self, arr):
        import torch

        with torch.no_grad():
            return self.train_forward(torch.as_tensor(np.asarray(arr, np.float32))).numpy()

    def forecast(self, horizon, inputs):
        self.calls.append(("point", len(inputs)))
        r = self._pred(np.stack(inputs))[:, :horizon]
        return r * 0.5, np.repeat(r[:, :, None], 10, axis=2)  # point = q50 (khác mean); quantile[..., 0] = mean

    def forecast_with_covariates(self, inputs, dynamic_numerical_covariates=None, **kw):
        assert len(inputs) == 1  # 1 origin/lời gọi (xreg fit beta_hat chung)
        self.calls.append(("cov", tuple(dynamic_numerical_covariates)))
        r = self._pred(np.stack(inputs))
        cov = np.array([v[0] for v in dynamic_numerical_covariates.values()])
        r = r + 1e-4 * float(cov[:, -1].sum())
        return r * 0.5, np.repeat(r[:, :, None], 10, axis=2)


LORA = {"targets": ("proj",), "r": 2, "alpha": 4.0, "max_epochs": 3, "patience": 2, "batch_size": 128, "lr": 1e-2, "train_stride": 4}


@pytest.fixture(autouse=True)
def _fresh_adapter_cache(monkeypatch):
    monkeypatch.setattr(models_tfm, "_ADAPTER_STATES", {})
    monkeypatch.setattr(models_tfm, "_ADAPTER_META", {})


def _model(tmp_path, stub=None, **kw):
    return TimesFMLoRAModel(device="cpu", allow_cpu=True, context=512, batch_size=64, model=stub or StubLoRATFM(),
                            lora=LORA, adapter_dir=str(tmp_path / "lora"), **kw)


def _spy_windows(monkeypatch):
    seen = []
    orig = TimesFMLoRAModel.windows

    def spy(self, seq, stride=1):
        seen.append(np.asarray(seq.idx).copy())
        return orig(self, seq, stride)

    monkeypatch.setattr(TimesFMLoRAModel, "windows", spy)
    return seen


# ----------------------------------------------------------------------------- (11) FIT/ES/VAL
def test_lora_trains_on_fit_selects_epoch_on_es_never_sees_val(tmp_path, store, folds, monkeypatch):
    seen = _spy_windows(monkeypatch)
    m = _model(tmp_path)
    f = folds[0]
    run = run_config(store, m, ColSet((), ()), [f], rounds=None, seed=1, keep_states=True)
    fit, es, val = (p.origins(store.ts, store.eligible) for p in (f.fit, f.es, f.val))
    assert len(seen) == 2 and np.array_equal(seen[0], fit) and np.array_equal(seen[1], es)  # FIT để học, ES để chọn epoch
    assert all(not np.intersect1d(s, val).size for s in seen)  # VAL không bao giờ vào training
    assert m.train_calls == 1 and 1 <= int(run.best_iters[0][0]) <= LORA["max_epochs"]
    files = sorted((tmp_path / "lora").glob("*.pt"))
    assert len(files) == 1 and files[0].with_suffix(".json").exists()
    meta = json.loads(files[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["mode"] == "es" and meta["n_windows_es"] > 0 and meta["sha256"] == state_sha256(lora_state_dict(m._wrappers()["module"]))
    assert meta["replaced_modules"] == ["proj"] and meta["n_trainable"] == 2 * (8 + 3)


def test_fixed_epochs_do_not_touch_es(tmp_path, store, folds, monkeypatch):
    seen = _spy_windows(monkeypatch)
    m = _model(tmp_path)
    run = run_config(store, m, ColSet((), ()), folds[:1], rounds=(2, 2, 2), seed=1, keep_states=False)
    assert len(seen) == 1 and np.array_equal(seen[0], folds[0].fit.origins(store.ts, store.eligible))  # không đọc ES
    assert tuple(run.best_iters[0]) == (2, 2, 2)
    meta = json.loads(next((tmp_path / "lora").glob("*ep2.json")).read_text(encoding="utf-8"))
    assert meta["mode"] == "fixed2" and meta["n_windows_es"] == 0


# ----------------------------------------------------------------------------- (12)(13) candidate không train lại, cùng adapter
def test_candidates_reuse_one_frozen_adapter(tmp_path, store, folds):
    m = _model(tmp_path)
    use = folds[:1]
    rounds = {f.name: (2, 2, 2) for f in use}
    base = ColSet((), ())
    base_run = run_config(store, m, base, use, rounds=rounds, seed=1, keep_states=False)
    assert m.train_calls == 1
    key = next(iter(models_tfm._ADAPTER_STATES))
    sha0 = models_tfm._ADAPTER_META[key]["sha256"]
    cands = [SHORT_BY_NAME[c] for c in SHORT_COLUMNS[:2]]
    lr = add_one_loop(store, m, base, base_run.rmse, cands, use, rounds, 0.5, 1, base_run.e0)
    assert m.train_calls == 1 and len(models_tfm._ADAPTER_STATES) == 1  # 2 candidate → 0 lần train thêm
    assert state_sha256(lora_state_dict(m._wrappers()["module"])) == sha0  # trọng số LoRA không đổi
    assert len(lr.table) == 2 and any(("cov", tuple(lr.final.ext)) == c[:2] for c in m._injected.calls if lr.final.ext)
    pruned, df = prune_pi(store, m, lr.final, use, rounds, seed=1, repeats=1)
    assert m.train_calls == 1 and state_sha256(lora_state_dict(m._wrappers()["module"])) == sha0
    # confirmation (ES bật, 2 seed) → 2 adapter mới; native confirmation dùng lại đúng 2 adapter đó
    conf = confirm(store, m, lr.final, use, (1, 2), keep_states=True)
    assert m.train_calls == 3 and len(conf.runs) == 2
    nat = confirm(store, m, ColSet((), ()), use, (1, 2), keep_states=True)
    assert m.train_calls == 3 and nat.rmse_mean.shape == (1, 3)


def test_adapter_reloads_from_disk_deterministically_and_freeze_is_asserted(tmp_path, store, folds, monkeypatch):
    m1 = _model(tmp_path)
    r1 = run_config(store, m1, ColSet((), ()), folds[:1], rounds=(2, 2, 2), seed=1, keep_states=True)
    monkeypatch.setattr(models_tfm, "_ADAPTER_STATES", {})
    monkeypatch.setattr(models_tfm, "_ADAPTER_META", {})
    m2 = _model(tmp_path)  # stub mới cùng init → cùng base; adapter phải tải từ .pt, không train
    r2 = run_config(store, m2, ColSet((), ()), folds[:1], rounds=(2, 2, 2), seed=1, keep_states=True)
    assert m2.train_calls == 0 and np.allclose(r1.states[0].yhat, r2.states[0].yhat, atol=1e-7)
    # đổi trọng số LoRA ngoài luồng → predict phải phát hiện (freeze bị vi phạm)
    import torch

    with torch.no_grad():
        m2._wrappers()["module"].proj.lora_B.add_(1.0)
    with pytest.raises(RuntimeError, match="freeze"):
        r2.states[0].result.predictors[0](r2.states[0].X_val)


def test_predictor_reloads_its_own_fold_adapter(tmp_path, store, folds):
    m = _model(tmp_path)
    run = run_config(store, m, ColSet((), ()), folds, rounds=(1, 1, 1), seed=1, keep_states=True)
    assert m.train_calls == len(folds)
    st0 = run.states[0]  # adapter đang nạp là của fold cuối → predictor fold 0 phải nạp lại adapter fold 0
    again = st0.result.predictors[0](st0.X_val)
    assert np.allclose(again, st0.yhat, atol=1e-7)


# ----------------------------------------------------------------------------- (14) tfm-final: +XReg vs native
def _cfg(tmp_path):
    return RunConfig(dataset_label="synthetic_tfm", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"], test_start="2026-01-04",
                     root=str(tmp_path), eval_seeds=(1, 2), selection_seed=1, models={"tfm": {"device": "cpu"}})


def _win(exp, name, ext, rmse, eps=0.02):
    (exp / "wins").mkdir(parents=True, exist_ok=True)
    (exp / "wins" / name).with_suffix(".json").write_text(json.dumps(
        {"model": name, "colset": {"b0": [], "ext": list(ext)}, "rmse_mean": [list(r) for r in rmse], "e0": [[100.0, 140.0, 170.0]],
         "eps": eps, "eval_seeds": [1, 2], "which": "prune", "median_gain_vs_e0": 0.1}), encoding="utf-8")
    np.savez_compressed(exp / "wins" / f"{name}_seed0.npz", idx_0=np.arange(3), yhat_0=np.zeros((3, 3), np.float32))


def _champion(exp):
    (exp / "champion.json").write_text(json.dumps({"model": "lgbm", "colset": {"b0": [], "ext": []}, "rmse_mean": [[95.0, 135.0, 165.0]],
                                                   "eps": 0.02, "e0": [[100.0, 140.0, 170.0]]}), encoding="utf-8")


@pytest.mark.parametrize("xreg_rmse,expect", [((80.0, 120.0, 150.0), "tfm_xreg"), ((89.99, 129.99, 159.99), "tfm_native"), ((91.0, 131.0, 161.0), "tfm_native")])
def test_tfm_final_compares_full_system_against_native_lora(tmp_path, monkeypatch, xreg_rmse, expect):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    exp.mkdir(parents=True)
    _champion(exp)
    _win(exp, "tfm_native", [], [(90.0, 130.0, 160.0)], eps=0.02)
    _win(exp, "tfm_xreg", ["ret_2", "rsi3_centered"], [xreg_rmse])
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    fin = json.loads((exp / "wins" / "tfm.json").read_text(encoding="utf-8"))
    assert fin["model"] == "tfm" and fin["role"] == "TimesFM-final" and fin["configuration"] == expect
    assert fin["compare_xreg_vs_native"]["eps"] == 0.02 and (exp / "wins" / "tfm_seed0.npz").exists()
    df = pd.read_csv(exp / "tfm_final.csv")
    assert set(df["configuration"]) == {"tfm_native", "tfm_xreg"} and df.loc[df["is_final"], "configuration"].iloc[0] == expect
    ch = pd.read_csv(exp / "champion_log.csv")
    assert (ch["model"] == "tfm").any() and not (ch["model"].isin(["tfm_xreg", "tfm_native"])).any()  # XReg không phải model độc lập


def test_tfm_final_requires_both_configs_and_a_champion(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    exp.mkdir(parents=True)
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    _win(exp, "tfm_native", [], [(90.0, 130.0, 160.0)])
    with pytest.raises(SystemExit) as e:
        cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "tfm_xreg" in str(e.value)
    _win(exp, "tfm_xreg", ["ret_2"], [(80.0, 120.0, 150.0)])
    with pytest.raises(SystemExit) as e:  # §3: champion ban đầu phải là LightGBM
        cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "loop --model lgbm" in str(e.value)


# ----------------------------------------------------------------------------- loop --model tfm end-to-end (stub)
def test_loop_tfm_end_to_end_with_stub(tmp_path, store, folds, monkeypatch):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    (exp / "s0").mkdir(parents=True)
    _champion(exp)
    ColSet((), ()).save(exp / "s0" / "tfm.json")
    (exp / "s0" / "candidates_tfm.json").write_text(json.dumps(
        {"model": "tfm", "candidates": list(SHORT_COLUMNS[:2]), "audit_dataset_label": cfg.dataset_label}), encoding="utf-8")
    # audit trên dataset khác → loop phải từ chối (collision audit phải chạy trên đúng data đang dùng)
    from p0.s0 import load_lock

    with pytest.raises(ValueError, match="lock-s0"):
        load_lock(exp, "tfm", dataset_label="other_dataset")
    use = folds[:1]
    stub = StubLoRATFM()
    model = _model(tmp_path, stub)
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_store", lambda c, **k: (store, use, None, None))
    monkeypatch.setattr(cli, "model_for", lambda c, name, allow_cpu: model)
    cli.cmd_loop(cfg, Namespace(model="tfm", smoke=True, allow_cpu=True, max_candidates=None, no_standalone=True, latency_origins=10, resume=False))
    for name in ("tfm_xreg", "tfm_native"):
        w = json.loads((exp / "wins" / f"{name}.json").read_text(encoding="utf-8"))
        assert w["eval_seeds"] == [1, 2] and (exp / "wins" / f"{name}_seed1.npz").exists()
    assert len(pd.read_csv(exp / f"keepdrop_tfm.csv")) == 2 and (exp / "calib" / "tfm_base.json").exists()
    assert not list((exp / "summary").glob("*.png")) if (exp / "summary").exists() else True  # không vẽ trong training
    ch = pd.read_csv(exp / "champion_log.csv")
    assert ch.loc[ch["model"] == "tfm", "decision"].str.startswith("probe").all()  # loop tfm chỉ ghi dòng probe, không so champion
    # số lần train = 1 calib(ES) + 2 eval seed (fixed epoch, seed1 = selection seed) + 2 confirmation (ES) = 5 — không phụ thuộc số candidate
    assert model.train_calls == 5
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert (exp / "wins" / "tfm.json").exists() and (pd.read_csv(exp / "champion_log.csv")["model"] == "tfm").any()
