"""TimesFM-LoRA (quyết định user 2026-09-03) — kiểm bằng stub thay package thật (timesfm chưa cài local):

11. LoRA: FIT chỉ để học, ES chỉ để chọn epoch, VAL không bao giờ vào training, adapter freeze trước candidate search;
12. thêm candidate KHÔNG train lại / KHÔNG đổi trọng số LoRA;   13. XReg search dùng CÙNG adapter đã freeze;
14. tfm-final: so HAI HỆ THỐNG HOÀN CHỈNH — A = TimesFM-LoRA baseline (feature-free) vs B = CÙNG adapter + XReg(F_win),
    theo luật project (> +ε_TFM) — KHÔNG phải "XReg vs LoRA" (XReg không phải model độc lập);
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


# ------------------------------------- (14) tfm-final: hệ thống B {LoRA + XReg(F_win)} vs hệ thống A {LoRA baseline}
def _cfg(tmp_path):
    return RunConfig(dataset_label="synthetic_tfm", hf_csv="data/hf.csv", lf_csv=None, val_days=["2026-01-03"], test_start="2026-01-04",
                     root=str(tmp_path), eval_seeds=(1, 2), selection_seed=1, models={"tfm": {"device": "cpu"}})


ADAPTERS = [{"key": "tfm_lora_fitA-fitB_esC-esD_seed1_es", "sha256": "abc123", "best_epoch": 2, "mode": "es"}]


def _win(exp, name, ext, rmse, eps=0.02, adapters=ADAPTERS, confirmation=True):
    (exp / "wins").mkdir(parents=True, exist_ok=True)
    m = TimesFMLoRAModel(device="cpu", allow_cpu=True, context=512, batch_size=64, model=StubLoRATFM(), lora=LORA)
    payload = {"model": name, "colset": {"b0": [], "ext": list(ext)}, "rmse_mean": [list(r) for r in rmse],
               "e0": [[100.0, 140.0, 170.0]], "eps": eps, "eval_seeds": [1, 2], "which": "prune", "median_gain_vs_e0": 0.1,
               "lora_adapters": adapters, **m.artifact_meta(ext, native=not ext)}
    if name == cli.TFM_XREG_WIN and confirmation:  # B: F_win đã thắng confirmation F_raw vs F_pruned
        payload["feature_set"] = "F_win"
        payload["feature_set_source"] = {"stage": "confirmation F_raw vs F_pruned (3 eval seed, ES bật)", "which": "prune",
                                         "n_new_raw": len(ext) + 1, "n_new_pruned": len(ext), "n_new_win": len(ext)}
    (exp / "wins" / name).with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
    np.savez_compressed(exp / "wins" / f"{name}_seed0.npz", idx_0=np.arange(3), yhat_0=np.zeros((3, 3), np.float32))


def _champion(exp):
    (exp / "champion.json").write_text(json.dumps({"model": "lgbm", "colset": {"b0": [], "ext": []}, "rmse_mean": [[95.0, 135.0, 165.0]],
                                                   "eps": 0.02, "e0": [[100.0, 140.0, 170.0]]}), encoding="utf-8")


@pytest.mark.parametrize("xreg_rmse,expect", [((80.0, 120.0, 150.0), "tfm_lora_xreg"), ((89.99, 129.99, 159.99), "tfm_lora_baseline"),
                                             ((91.0, 131.0, 161.0), "tfm_lora_baseline")])
def test_tfm_final_compares_two_complete_systems(tmp_path, monkeypatch, xreg_rmse, expect):
    """(6) tfm-final chọn giữa HAI HỆ THỐNG HOÀN CHỈNH: A = LoRA baseline feature-free, B = CÙNG LoRA + XReg(F_win)."""
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    exp.mkdir(parents=True)
    _champion(exp)
    _win(exp, cli.TFM_BASELINE_WIN, [], [(90.0, 130.0, 160.0)], eps=0.02)
    _win(exp, cli.TFM_XREG_WIN, ["ret_2", "rsi3_centered"], [xreg_rmse])
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    fin = json.loads((exp / "wins" / "tfm.json").read_text(encoding="utf-8"))
    assert fin["model"] == "tfm" and fin["role"] == "TimesFM-final" and fin["configuration"] == expect
    assert fin["compare_systems"]["A"] == cli.TFM_BASELINE_WIN and fin["compare_systems"]["B"] == cli.TFM_XREG_WIN
    assert fin["compare_systems"]["eps"] == 0.02 and (exp / "wins" / "tfm_seed0.npz").exists()
    # (30) metadata tường minh: LoRA, baseline hay +XReg, covariates, chuỗi vào/ra
    assert fin["finetune_method"] == "LoRA" and fin["backbone"] == "timesfm-2.5-200m" and fin["input_series"] == "btc_1m_log_return"
    assert fin["native"] == (expect == cli.TFM_BASELINE_WIN)
    assert fin["covariates"] == ([] if expect == cli.TFM_BASELINE_WIN else ["ret_2", "rsi3_centered"])
    assert fin["target"] == "cumulative_log_return_y1_y2_y3" and fin["context"] == 512 and fin["forecast_horizon"] == 3 and fin["colset"]["b0"] == []
    df = pd.read_csv(exp / "tfm_final.csv")
    assert set(df["configuration"]) == {cli.TFM_BASELINE_WIN, cli.TFM_XREG_WIN} and set(df["system"]) == {"A", "B"}
    assert df.loc[df["is_final"], "configuration"].iloc[0] == expect
    ch = pd.read_csv(exp / "champion_log.csv")
    # (7)(8) chỉ TFM-final vào champion; cấu hình nội bộ không bao giờ
    assert (ch["model"] == "tfm").any() and not ch["model"].isin(list(cli.CHAMPION_INELIGIBLE)).any()


def test_tfm_final_reads_legacy_baseline_name(tmp_path, monkeypatch):
    """Artifact tên cũ `tfm_lora_native.json` vẫn đọc được (ngữ nghĩa không đổi: hệ thống A)."""
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    exp.mkdir(parents=True)
    _champion(exp)
    _win(exp, cli.TFM_BASELINE_LEGACY, [], [(90.0, 130.0, 160.0)], eps=0.02)
    _win(exp, cli.TFM_XREG_WIN, ["ret_2"], [(80.0, 120.0, 150.0)])
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    fin = json.loads((exp / "wins" / "tfm.json").read_text(encoding="utf-8"))
    assert fin["baseline_artifact"] == cli.TFM_BASELINE_LEGACY and fin["configuration"] == cli.TFM_XREG_WIN


def test_tfm_final_refuses_when_f_win_not_confirmed_or_adapter_differs(tmp_path, monkeypatch):
    """(1) F_win phải đến TỪ confirmation raw-vs-pruned; (5) hai hệ thống phải cùng adapter LoRA đã freeze."""
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    exp.mkdir(parents=True)
    _champion(exp)
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    _win(exp, cli.TFM_BASELINE_WIN, [], [(90.0, 130.0, 160.0)])
    _win(exp, cli.TFM_XREG_WIN, ["ret_2"], [(80.0, 120.0, 150.0)], confirmation=False)
    w = json.loads((exp / "wins" / f"{cli.TFM_XREG_WIN}.json").read_text(encoding="utf-8"))
    w.pop("which")  # không còn bằng chứng nào của confirmation
    (exp / "wins" / f"{cli.TFM_XREG_WIN}.json").write_text(json.dumps(w), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "TFM_FLOW_ORDER" in str(e.value)
    _win(exp, cli.TFM_XREG_WIN, ["ret_2"], [(80.0, 120.0, 150.0)],
         adapters=[{"key": "khac", "sha256": "zzz", "best_epoch": 1, "mode": "es"}])
    with pytest.raises(SystemExit) as e:
        cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "TFM_ADAPTER_IDENTITY" in str(e.value)


def test_tfm_final_requires_both_configs_and_a_champion(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    exp.mkdir(parents=True)
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    _win(exp, cli.TFM_BASELINE_WIN, [], [(90.0, 130.0, 160.0)])
    with pytest.raises(SystemExit) as e:
        cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert "tfm_lora_xreg" in str(e.value)
    _win(exp, cli.TFM_XREG_WIN, ["ret_2"], [(80.0, 120.0, 150.0)])
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
    wins = {}
    for name in (cli.TFM_XREG_WIN, cli.TFM_BASELINE_WIN):
        w = wins[name] = json.loads((exp / "wins" / f"{name}.json").read_text(encoding="utf-8"))
        assert w["eval_seeds"] == [1, 2] and (exp / "wins" / f"{name}_seed1.npz").exists()
        assert w["finetune_method"] == "LoRA" and w["colset"]["b0"] == [] and w["configuration"] == name  # (22)(23)(30)
        assert w["native"] == (len(w["covariates"]) == 0)
    # (3) hệ thống A: 0 B0, 0 covariate;  (2)(4) hệ thống B: CÙNG LoRA + XReg đúng F_win đã thắng confirmation
    A, B = wins[cli.TFM_BASELINE_WIN], wins[cli.TFM_XREG_WIN]
    assert A["system"] == "A" and A["native"] and A["colset"]["ext"] == [] and A["colset"]["b0"] == []
    assert B["system"] == "B" and B["feature_set"] == "F_win" and B["feature_set_source"]["which"] in ("prune", "unprune")
    assert B["colset"]["ext"] == B["covariates"] and B["feature_set_source"]["n_new_win"] == len(B["colset"]["ext"])
    # (5) hai hệ thống dùng ĐÚNG một bộ adapter LoRA đã freeze
    assert A["lora_adapters"] and A["lora_adapters"] == B["lora_adapters"]
    log = pd.read_csv(exp / "log.csv")
    conf_rows = log[log["step"] == "confirm"].reset_index()
    i_base = conf_rows.index[conf_rows["colset"] == "baseline"]
    i_raw = conf_rows.index[conf_rows["colset"].isin(["unprune", "prune"])]
    assert len(i_raw) and (not len(i_base) or i_base.min() > i_raw.max())  # (1) raw vs pruned XONG rồi mới dựng baseline
    assert (log.loc[log["step"] == "calibrate", "note"].str.contains(r"LoRA FIT \+ ES")).any()  # (24) calibrate = LoRA FIT + ES
    assert len(pd.read_csv(exp / f"keepdrop_tfm.csv")) == 2 and (exp / "calib" / "tfm_base.json").exists()
    assert not list((exp / "summary").glob("*.png")) if (exp / "summary").exists() else True  # không vẽ trong training
    ch = pd.read_csv(exp / "champion_log.csv")
    assert ch.loc[ch["model"] == "tfm", "decision"].str.startswith("probe").all()  # loop tfm chỉ ghi dòng probe, không so champion
    assert not ch["model"].isin(list(cli.CHAMPION_INELIGIBLE)).any()  # (8) artifact nội bộ không bao giờ đụng champion
    # số lần train = 1 calib(ES) + 2 eval seed (fixed epoch, seed1 = selection seed) + 2 confirmation (ES) = 5 — không phụ thuộc số candidate
    assert model.train_calls == 5
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    assert (exp / "wins" / "tfm.json").exists() and (pd.read_csv(exp / "champion_log.csv")["model"] == "tfm").any()


def test_new_tfm_baseline_is_lora_native_and_b0_never_enters_xreg(tmp_path, store, folds):
    """(21)(22)(23): `tfm` = TimesFMLoRAModel (không zero-shot), S0 = ∅, cột B0 trong colset không bao giờ thành covariate."""
    from p0.s0 import s0_for

    m = cli.make_model("tfm", {"device": "cpu"}, allow_cpu=True)
    assert isinstance(m, TimesFMLoRAModel) and m.series_covariates == "ext" and m.supports_rounds and m.seed_dependent
    assert s0_for("tfm", None) == ColSet((), ())
    stub = _model(tmp_path)
    st = run_config(store, stub, ColSet(store.b0_names[:5], ("ret_2",)), folds[:1], rounds=(1, 1, 1), seed=1, keep_states=True).states[0]
    assert st.X_val.cov_names == ("ret_2",) and st.X_val.cov.shape[1] == 1  # B0* KHÔNG được đưa vào XReg
    st0 = run_config(store, stub, ColSet(store.b0_names[:5]), folds[:1], rounds=(1, 1, 1), seed=1, keep_states=True).states[0]
    assert st0.X_val.cov is None  # chỉ B0 → native (không covariate)


# ------------------------------- (6)(7)(8) confirmation chấm HAI HỆ THỐNG HOÀN CHỈNH, cùng adapter đã freeze
def test_confirmation_scores_complete_systems_not_bare_xreg(tmp_path, store, folds, monkeypatch):
    """F_raw và F_pruned được chấm dưới dạng {TimesFM-LoRA + XReg(F)} trên CÙNG adapter — không có
    "XReg vs XReg" và không có XReg đứng một mình."""
    cfg = _cfg(tmp_path)
    exp = cfg.exp_dir
    (exp / "s0").mkdir(parents=True)
    _champion(exp)
    ColSet((), ()).save(exp / "s0" / "tfm.json")
    (exp / "s0" / "candidates_tfm.json").write_text(json.dumps(
        {"model": "tfm", "candidates": list(SHORT_COLUMNS[:2]), "audit_dataset_label": cfg.dataset_label}), encoding="utf-8")
    model = _model(tmp_path, StubLoRATFM())
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_store", lambda c, **k: (store, folds[:1], None, None))
    monkeypatch.setattr(cli, "model_for", lambda c, name, allow_cpu: model)
    cli.cmd_loop(cfg, Namespace(model="tfm", smoke=True, allow_cpu=True, max_candidates=None, no_standalone=True,
                                latency_origins=None, resume=False))
    # (6) bảng prune ghi rõ hệ thống của TỪNG bên
    df = pd.read_csv(exp / "prune_tfm.csv")
    assert list(df["system"]) == ["TimesFM-LoRA + XReg(F_raw)", "TimesFM-LoRA + XReg(F_pruned)"]
    src = json.loads((exp / "wins" / f"{cli.TFM_XREG_WIN}.json").read_text(encoding="utf-8"))["feature_set_source"]
    assert src["compared_systems"] == ["TimesFM-LoRA + XReg(F_raw)", "TimesFM-LoRA + XReg(F_pruned)"]
    # (8) hai bên confirmation dùng CÙNG adapter LoRA đã freeze
    assert src["same_frozen_lora_adapter"] is True
    xr = json.loads((exp / "wins" / f"{cli.TFM_XREG_WIN}.json").read_text(encoding="utf-8"))
    base = json.loads((exp / "wins" / f"{cli.TFM_BASELINE_WIN}.json").read_text(encoding="utf-8"))
    assert xr["lora_adapters"] and xr["lora_adapters"] == base["lora_adapters"]
    assert xr["scored_system_per_candidate"].startswith("TimesFM-LoRA + XReg(")
    # (7) không nơi nào mô tả quyết định là "XReg vs XReg" / "XReg vs LoRA"
    for f in (exp / "prune_tfm.csv", exp / "wins" / f"{cli.TFM_XREG_WIN}.json", exp / "wins" / f"{cli.TFM_BASELINE_WIN}.json"):
        txt = f.read_text(encoding="utf-8").lower()
        assert "xreg vs xreg" not in txt and "xreg vs lora" not in txt and "xreg vs native" not in txt


def test_official_docs_never_say_xreg_vs_xreg():
    """(7) mô tả chính thức (code + doc đang hiệu lực) không được dùng cách gọi sai."""
    import pathlib as _p

    root = _p.Path(__file__).resolve().parents[1]
    files = list((root / "src" / "p0").glob("*.py")) + [root / "README.md", root / ".claude" / "CLAUDE.md",
                                                        root / ".claude" / "AGENT.md", root / "docs" / "VAST_SESSION_PROMPT.md"]
    files += list((root / ".claude" / "agents").glob("*.md"))
    negation = ("không phải", "không bao giờ", "không gọi", "cấm", "sai", "never", "not ", "đừng")
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").lower().splitlines(), 1):
            for bad in ("xreg vs xreg", "xreg vs lora", "xreg vs timesfm"):
                if bad in line and not any(n in line for n in negation):  # chỉ chấp nhận khi câu đang PHỦ ĐỊNH cách gọi đó
                    raise AssertionError(f"{f}:{i} còn mô tả sai {bad!r}: {line.strip()[:120]}")


# ------------------------------- (13)(15) TFM-final LƯU rồi CHỜ; chỉ được so champion ở replay
def test_tfm_final_saves_representative_and_waits_for_replay(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    object.__setattr__(cfg, "defer_champion", True)
    exp = cfg.exp_dir
    exp.mkdir(parents=True)
    _champion(exp)
    _win(exp, cli.TFM_BASELINE_WIN, [], [(90.0, 130.0, 160.0)], eps=0.02)
    _win(exp, cli.TFM_XREG_WIN, ["ret_2"], [(80.0, 120.0, 150.0)])
    monkeypatch.setattr(cli, "gate", lambda *a, **k: None)
    cli.cmd_tfm_final(cfg, Namespace(smoke=True, allow_cpu=True))
    fin = json.loads((exp / "wins" / "tfm.json").read_text(encoding="utf-8"))
    assert fin["representative"] == "tfm" and fin["configuration"] == cli.TFM_XREG_WIN  # (13) đã LƯU đại diện
    ch = pd.read_csv(exp / "champion_log.csv") if (exp / "champion_log.csv").exists() else None
    assert ch is None or not (ch["model"] == "tfm").any()  # (13) CHƯA so champion
    assert json.loads((exp / "champion.json").read_text(encoding="utf-8"))["model"] == "lgbm"  # champion chưa đổi
    # (15) đại diện TimesFM ĐỦ TƯ CÁCH champion ở bước replay
    (exp / "champion.json").unlink()
    for m, rmse in (("lgbm", [[95.0, 135.0, 165.0]]), ("xgb", [[96.0, 136.0, 166.0]])):
        (exp / "wins" / f"{m}.json").write_text(json.dumps(
            {"model": m, "colset": {"b0": [], "ext": []}, "rmse_mean": rmse, "e0": [[100.0, 140.0, 170.0]], "eps": 0.02,
             "eval_seeds": [1, 2], "which": "prune", "median_gain_vs_e0": 0.3, "champion_extra": {"win": "prune"}}), encoding="utf-8")
    object.__setattr__(cfg, "model_order", ["lgbm", "xgb", "tfm"])
    cli.cmd_champion_replay(cfg, Namespace(allow_partial=False, force_replay=False))
    ch = pd.read_csv(exp / "champion_log.csv")
    order = [m for m in ch["model"].tolist() if m in ("lgbm", "xgb", "tfm")]
    assert order == ["lgbm", "xgb", "tfm"]  # thứ tự methodology cố định, tfm là đại diện hợp lệ
    assert json.loads((exp / "champion.json").read_text(encoding="utf-8"))["model"] == "tfm"  # RMSE tfm tốt nhất
