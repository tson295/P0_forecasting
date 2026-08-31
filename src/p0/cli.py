"""CLI: python run.py <step> --config configs/p0_15d.json [--model lgbm] [--smoke] [--allow-cpu]

Bước (§8): check-data → calibrate (lgbm, b0306) → filter-b0 → loop --model m (calibrate riêng trên B0*, add-one, prune PI,
3 seed → win_m, latency, champion + figure) → tfm-final (§2.2 #4: chọn giữa hai nhánh TimesFM) →
autots-search (§2.2 #6: framework AutoTS trên F_WR_best và F_MR_best → AutoTS-final) → ensemble → final. `smoke-e2e` chạy toàn bộ trên data tổng hợp CPU (chỉ debug).
Ba vai trò seed tách bạch (§1.3): `calib_seed` (seed0) CHỈ cho run ES tìm số vòng cố định; `eval_seeds` (seed1/2/3) đo ε
và chạy confirmation 3 seed; `selection_seed` là MỘT seed cố định cho mọi bước selection (R1–R4, baseline + 39 candidate,
prune PI) để chênh lệch RMSE chỉ đến từ feature set.

Gate: training chỉ khi `.claude/MEMORY.md` ghi `TRAINING: UNLOCKED`; GPU preflight bắt buộc. `--smoke` / `--allow-cpu` CHỈ được chấp nhận
khi dataset_label bắt đầu bằng "synthetic" (data tổng hợp) — với data thật CLI từ chối (plan §0: cấm training CPU).
Data thật phải khớp `data/data_checksums.json` (§6.1) ở mọi bước sau check-data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import HORIZONS, RunConfig
from .data import check_ohlcv, read_ohlcv_csv, verify_checksums, write_checksums
from .features_ext import CANDIDATE_BY_NAME, CANDIDATES
from .filter_b0 import FilterTable, median_over_folds, mutual_info, permutation_importance, standalone_gain, verify_sets
from .harness import ColSet, Store, calibrate, rounds_from, run_at_seed, run_config, seed_noise
from .latency import measure_tabular
from .logs import load_preds, log_champion, log_latency, log_run, new_exp_id, save_run
from .loop import (add_one_loop, compare, confirm, decide_win, ensemble_rmse, inverse_mse_weights, load_champion, prune_pi,
                   save_champion)
from .metrics import cell_metrics, e0_rmse, gain_pp, mean_rmse_over_seeds, seed_noise_cells, seed_noise_eps, summarize
from .models import make_model
from .palette import LABEL
from .split import Partition, check_fold, make_final, make_folds, utc_ts


# ----------------------------------------------------------------------------- helpers
def say(msg: str) -> None:
    print(f"[p0 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def training_state(root: Path) -> str:
    mem = root / ".claude" / "MEMORY.md"
    if not mem.exists():
        return "UNKNOWN"
    for line in mem.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]:
        if line.startswith("TRAINING:"):
            return line.split(":", 1)[1].strip().upper()
    return "UNKNOWN"


def _is_synthetic(cfg: RunConfig) -> bool:
    return str(cfg.dataset_label).startswith("synthetic")


def gate(cfg: RunConfig, args, model_names: list[str]) -> None:
    """Khóa training (MEMORY) + GPU preflight. --smoke / --allow-cpu CHỈ hợp lệ với dataset tổng hợp (dataset_label 'synthetic*')."""
    smoke, allow_cpu = bool(getattr(args, "smoke", False)), bool(getattr(args, "allow_cpu", False))
    if (smoke or allow_cpu) and not _is_synthetic(cfg):
        sys.exit(f"--smoke/--allow-cpu bị từ chối với dataset '{cfg.dataset_label}': chỉ cho data tổng hợp "
                 "(plan §0: cấm training CPU; không bỏ khóa training/GPU gate trên data thật).")
    if smoke:
        return
    st = training_state(Path(cfg.root))
    if st != "UNLOCKED":
        sys.exit(f"TRAINING_LOCKED (MEMORY.md: {st}) — cần user unlock rõ ràng trước khi chạy training.")
    if cfg.require_gpu and not allow_cpu:
        for m in model_names:
            gpu_preflight(m, cfg)


def gpu_preflight(model: str, cfg: RunConfig) -> None:
    if model == "lgbm":
        from Baseline_LGBM import LGBMConfig, assert_p100_lightgbm

        assert_p100_lightgbm(LGBMConfig(require_p100=False, **cfg.model_params("lgbm")))
        say("GPU preflight LightGBM: OK")
    elif model in ("xgb", "xgbrf"):
        import xgboost as xgb

        x = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
        xgb.XGBRegressor(n_estimators=3, device="cuda", tree_method="hist").fit(x, x[:, 0])
        say("GPU preflight XGBoost: OK")
    elif model == "cat":
        from catboost import CatBoostRegressor

        x = np.random.default_rng(0).normal(size=(256, 4))
        CatBoostRegressor(iterations=3, task_type="GPU", verbose=False, allow_writing_files=False).fit(x, x[:, 0])
        say("GPU preflight CatBoost: OK")
    elif model == "lstm" or model.startswith("tfm"):
        import torch

        if not torch.cuda.is_available():
            sys.exit(f"GPU preflight {model}: CUDA không có — cấm training/inference CPU.")
        say(f"GPU preflight torch ({model}): {torch.cuda.get_device_name(0)}")
        if model == "tfm":
            import timesfm  # noqa: F401  — chỉ kiểm tra đã cài đúng version (pin trong requirements.txt)
    elif model in ("autots_wr", "autots_mr"):
        import autots  # noqa: F401

        gpu_preflight("lgbm" if model == "autots_wr" else "xgb", cfg)  # regression_model bên trong chạy GPU
        say(f"GPU preflight AutoTS ({autots.__version__}): OK")


def checksum_path(cfg: RunConfig) -> Path:
    return Path(cfg.root) / "data" / "data_checksums.json"


def load_store(cfg: RunConfig, need_lf: bool = True, verify: bool = True):
    """Đọc data + kiểm tra §1.1; verify=True → sha256 phải khớp data/data_checksums.json (§6.1, trừ lúc check-data ghi file đó)."""
    if verify:
        ck = checksum_path(cfg)
        if not ck.exists():
            sys.exit(f"Thiếu {ck} — chạy `python run.py check-data --config <cfg> --write-checksums` (§6.1) trước.")
        ok, problems = verify_checksums(ck, Path(cfg.root), label=cfg.dataset_label)
        if not ok:
            sys.exit("Checksum data không khớp §6.1 — dừng: " + "; ".join(problems))
    hf_path = cfg.path(cfg.hf_csv)
    raw_hf = read_ohlcv_csv(hf_path)
    rep = check_ohlcv(raw_hf)
    if not rep["ok"]:
        sys.exit(f"Data HF không đạt §1.1: {rep}")
    raw_lf = None
    lf_path = cfg.path(cfg.lf_csv) if cfg.lf_csv else None
    if need_lf and lf_path and lf_path.exists():
        raw_lf = read_ohlcv_csv(lf_path)
    store = Store(raw_hf, raw_lf)
    folds = make_folds(store.first_origin_ts, cfg.val_days, cfg.purge_minutes, cfg.es_hours)
    test_end = utc_ts(cfg.test_end) if cfg.test_end else store.last_ts + 60
    final = make_final(store.first_origin_ts, cfg.test_start, test_end, cfg.purge_minutes)
    return store, folds, final, rep


def _params_for(cfg: RunConfig, name: str) -> dict:
    """Params của model; hai nhánh TimesFM dùng chung khai báo `models.tfm` nếu không khai riêng."""
    p = cfg.model_params(name)
    if not p and name.startswith("tfm"):
        p = cfg.model_params("tfm")
    return dict(p)


def model_for(cfg: RunConfig, name: str, allow_cpu: bool):
    if allow_cpu and not _is_synthetic(cfg):
        sys.exit(f"allow_cpu với dataset '{cfg.dataset_label}' bị cấm (plan §0: training chỉ GPU).")
    params = _params_for(cfg, name)
    if allow_cpu:  # smoke/unit: ép CPU rõ ràng
        params = {k: v for k, v in params.items() if k not in ("device_type", "device", "task_type")}
        # mặc định ép `device="cpu"` cho mọi model (xgb, xgbrf, lstm, tfm/tfm_b0/tfm_ext, autots_*); lgbm/cat dùng khoá riêng
        params.update({"lgbm": {"device_type": "cpu"}, "cat": {"task_type": "CPU"}}.get(name, {"device": "cpu"}))
    return make_model(name, params, allow_cpu=allow_cpu)


PROBE_MODELS = ("autots_wr", "autots_mr", "tfm_b0", "tfm_ext")  # model CHỈ để dò feature: chạy đủ §2.1
# (add-one → prune PI → confirmation) nhưng KHÔNG so champion, KHÔNG vào ensemble, KHÔNG refit ở Final.
# Bước chọn "final" tương ứng gộp kết quả của chúng thành một model đại diện:
FINAL_STEP = {"autots_wr": "autots-search", "autots_mr": "autots-search", "tfm_b0": "tfm-final", "tfm_ext": "tfm-final"}
TFM_BRANCH_BASE = {"tfm_ext": "empty", "tfm_b0": "b0star"}  # điểm xuất phát S của từng nhánh TimesFM


def colset_from_arg(store: Store, cfg: RunConfig, arg: str) -> ColSet:
    if arg == "b0306":
        return store.all_b0()
    if arg == "b0star":
        p = cfg.exp_dir / "b0_star.json"
        if not p.exists():
            sys.exit("Chưa có experiments/b0_star.json — chạy filter-b0 trước (§1.4).")
        return ColSet.load(p)
    return ColSet.load(Path(arg))


def candidates_from(cfg: RunConfig, limit: int | None) -> list:
    names = cfg.candidates or [c.name for c in CANDIDATES]
    cands = [CANDIDATE_BY_NAME[n] for n in names]
    return cands[:limit] if limit else cands


def _log(cfg: RunConfig, **row) -> None:
    row.setdefault("dataset_label", cfg.dataset_label)
    row.setdefault("config_hash", cfg.hash())
    log_run(cfg.exp_dir, row)


def _summ_row(run, base_rmse, base_name: str) -> dict:
    g = run.gain_vs(base_rmse)
    s = summarize(g)
    return {"base": base_name, **{k: round(s[k], 4) for k in ("MedianGain", "WinRate", "P10Gain", "WorstGain")},
            "rmse_cells": json.dumps(np.round(run.rmse, 3).tolist()), "mae_cells": json.dumps(np.round(run.mae, 3).tolist()),
            "e0_cells": json.dumps(np.round(run.e0, 3).tolist()), "gain_cells": json.dumps(np.round(g, 4).tolist()),
            "n_cols": len(run.colset.names)}


def _cells(a) -> str:
    return json.dumps(np.round(np.asarray(a, float), 4).tolist())


# ----------------------------------------------------------------------------- steps
def cmd_check_data(cfg: RunConfig, args) -> None:
    hf_path = cfg.path(cfg.hf_csv)
    raw = read_ohlcv_csv(hf_path)
    rep = check_ohlcv(raw)
    say(f"HF {hf_path}: {json.dumps(rep, ensure_ascii=False)}")
    reports = {"hf": rep}
    files = {"hf": hf_path}
    lf_path = cfg.path(cfg.lf_csv) if cfg.lf_csv else None
    if lf_path and lf_path.exists():
        lf = read_ohlcv_csv(lf_path)
        reports["lf"] = check_ohlcv(lf, step=300)
        files["lf"] = lf_path
        say(f"LF {lf_path}: {json.dumps(reports['lf'], ensure_ascii=False)}")
    store, folds, final, _ = load_store(cfg, verify=False)
    say(f"B0-eligible origins: {int(store.eligible.sum())} | first {pd.Timestamp(store.first_origin_ts, unit='s', tz='UTC')} | last {pd.Timestamp(store.last_ts, unit='s', tz='UTC')}")
    for f in folds + [final]:
        chk = check_fold(f, store.ts, store.eligible, cfg.purge_minutes)
        say(f"{f.name}: FIT {f.fit.label()} n={chk['n_fit']} | ES {f.es.label()} n={chk['n_es']} | VAL {f.val.label()} n={chk['n_val']} "
            f"| cuối=T_end−4' {chk['last_val_origin_is_Tend_minus_4min']} | {'OK' if chk['ok'] else chk['problems']}")
        if not chk["ok"]:
            sys.exit("Fold không đạt §1.2 — dừng.")
    out = checksum_path(cfg)
    if args.write_checksums:
        write_checksums(cfg.dataset_label, files, reports, out, root=Path(cfg.root))
        say(f"checksum (sha256, path tương đối root) → {out}")
    elif out.exists():
        ok, problems = verify_checksums(out, Path(cfg.root), label=cfg.dataset_label)
        say(f"verify {out}: {'OK — khớp snapshot đã ghi' if ok else 'KHÔNG KHỚP: ' + '; '.join(problems)}")
        if not ok:
            sys.exit("Data khác snapshot đã ghi (§6.1) — dừng.")
    else:
        say(f"chưa có {out} — chạy lại với --write-checksums để ghi anchor §6.1 (bắt buộc trước mọi bước training)")


def cmd_calibrate(cfg: RunConfig, args) -> dict:
    """§1.3: run ES → 15fixed_m; 3 seed số vòng cố định → ε_m. Lưu experiments/calib/<model>_<tag>.json."""
    gate(cfg, args, [args.model])
    store, folds, _, _ = load_store(cfg)
    colset = colset_from_arg(store, cfg, args.colset)
    model = model_for(cfg, args.model, args.allow_cpu)
    say(f"calibrate {args.model} trên {args.colset} ({len(colset.names)} cột) — ES một lần (calib_seed = {cfg.calib_seed})")
    cal = calibrate(store, model, colset, folds, seed=cfg.calib_seed, keep_states=False)
    rounds = rounds_from(cal) if getattr(model, "supports_rounds", True) else None
    say(f"số vòng cố định: {rounds}")
    say(f"ε: {len(cfg.eval_seeds)} evaluation seed {list(cfg.eval_seeds)} với số vòng cố định (không seed nào làm mốc)")
    eps, noise, runs = seed_noise(store, model, colset, folds, rounds, cfg.eval_seeds, cfg.eps_floor_pp)
    base = run_at_seed(runs, cfg.sel_seed)  # bảng RMSE mốc của selection (cùng seed với mọi bước selection)
    if base is None:
        base = run_config(store, model, colset, folds, rounds=rounds, seed=cfg.sel_seed, keep_states=False)
    out = {"model": args.model, "tag": args.colset, "colset": colset.to_dict(), "rounds": rounds, "eps": eps,
           "noise_cells": np.round(noise, 5).tolist(), "calib_seed": cfg.calib_seed, "eval_seeds": list(cfg.eval_seeds),
           "selection_seed": cfg.sel_seed, "rmse": base.rmse.tolist(), "e0": base.e0.tolist(),
           "best_iters_es": cal.best_iters.tolist(), "folds": cal.fold_names,
           "seed_rmse": [r.rmse.tolist() for r in runs], "config_hash": cfg.hash()}
    path = cfg.exp_dir / "calib" / f"{args.model}_{args.colset.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    exp_id = new_exp_id("calibrate", args.model, args.colset)
    _log(cfg, exp_id=exp_id, step="calibrate", model=args.model, seed=cfg.sel_seed, colset=args.colset, rounds=json.dumps(rounds),
         **_summ_row(base, base.e0, "E0"), decision=f"eps={eps:.4f}", train_device=getattr(model, "train_device", ""),
         note=f"calib_seed={cfg.calib_seed} eval_seeds={list(cfg.eval_seeds)} noise_rms={eps:.4f}")
    save_run(cfg.exp_dir, exp_id, {**base.to_dict(), "step": "calibrate", "tag": args.colset, "eps": eps, "rounds_fixed": rounds,
                                   "best_iters_es": cal.best_iters.tolist(), "seed_rmse": out["seed_rmse"], "config_hash": cfg.hash()})
    say(f"ε_{args.model} = {eps:.4f} pp (RMS nhiễu 15 ô; min {noise.min():.4f}, max {noise.max():.4f}) → {path}")
    return out


def _load_calib(cfg: RunConfig, model: str, tag: str) -> dict:
    p = cfg.exp_dir / "calib" / f"{model}_{tag}.json"
    if not p.exists():
        sys.exit(f"Thiếu {p} — chạy: python run.py calibrate --model {model} --colset {tag}")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_filter_b0(cfg: RunConfig, args) -> None:
    """§1.4: PI + SA + MI trên B0-306 → cờ ≥ 2/3 → R1–R4 → 4 run kiểm chứng (15fixed_306) → B0*."""
    gate(cfg, args, ["lgbm"])
    store, folds, _, _ = load_store(cfg)
    cal = _load_calib(cfg, "lgbm", "b0306")
    rounds = {k: tuple(v) for k, v in cal["rounds"].items()}
    eps = float(cal["eps"])
    model = model_for(cfg, "lgbm", args.allow_cpu)
    names = list(store.b0_names)
    if args.max_cols:
        names = names[: args.max_cols]
    colset = ColSet(tuple(names))
    say(f"(a) PI: run baseline B0-306 (15fixed_306, selection_seed {cfg.sel_seed}) giữ state rồi xáo {len(names)} cột × 3 lần trong VAL")
    base_run = run_config(store, model, colset, folds, rounds=rounds, seed=cfg.sel_seed, keep_states=True)
    base_rmse = base_run.rmse  # cùng một run làm mốc cho PI và cho 4 run kiểm chứng R1–R4
    if len(names) == len(store.b0_names) and "rmse" in cal:  # chỉ so được khi không cắt cột (--max-cols của smoke)
        d_cal = float(np.max(np.abs(base_rmse / np.asarray(cal["rmse"]) - 1)))
        say(f"   |RMSE base − RMSE calibrate| tối đa {d_cal * 100:.4f}% (cùng config + selection_seed → phải ≈ 0)")
    pi = median_over_folds(permutation_importance(store, base_run, list(range(len(names))), repeats=3, seed=cfg.sel_seed))
    say(f"(b) SA: {len(names)} model 1 cột (ES, selection_seed) — vs E0 và vs B0-306")
    sa_e0, sa_b0 = standalone_gain(store, model, folds, names, cfg.sel_seed, base_rmse,
                                   progress=lambda k, n, nm: say(f"  SA {k}/{n} {nm}") if k % 25 == 0 else None)
    say("(c) MI trên FIT (null xáo trộn)")
    mi = mutual_info(store, folds, colset, seed=cfg.sel_seed)
    table = FilterTable(names, pi, sa_e0, sa_b0, mi)
    df = table.to_frame()
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.exp_dir / "b0_filter.csv", index=False)
    red = df[(df[[f"SA_gain_b0306_h{h}" for h in (1, 2, 3)]] > eps).sum(axis=1) >= 2]
    if len(red):
        say(f"CỜ ĐỎ: {len(red)} cột đơn thắng B0-306 (≥ 2/3 horizon) — B0 bị nhiễu chi phối; R3/R4 sẽ tự thắng ở kiểm chứng")
    sets = table.sets()
    say("kiểm chứng R1–R4 (15fixed_306): " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))
    vdf, chosen, runs = verify_sets(store, model, sets, folds, rounds, base_rmse, eps, seed=cfg.sel_seed)
    vdf.to_csv(cfg.exp_dir / "b0_sets.csv", index=False)
    star = ColSet(tuple(sets[chosen])) if chosen in sets else colset
    star.save(cfg.exp_dir / "b0_star.json")
    for _, r in vdf.iterrows():
        exp_id = new_exp_id("filter_b0", "lgbm", str(r["set"]))
        extra = _summ_row(runs[r["set"]], base_rmse, "B0-306") if r["set"] in runs else {"base": "B0-306"}
        _log(cfg, exp_id=exp_id, step="filter_b0", model="lgbm", seed=cfg.sel_seed, colset=str(r["set"]), rounds="15fixed_306", **extra,
             decision="B0*" if r["chosen"] else "", note=f"n_cols={r['n_cols']}", train_device=getattr(model, "train_device", ""))
        if r["set"] in runs:
            save_run(cfg.exp_dir, exp_id, {**runs[r["set"]].to_dict(), "step": "filter_b0", "set": str(r["set"]), "base": "B0-306", "eps": eps})
    say(f"B0* = {chosen} ({len(star.names)} cột) → experiments/b0_star.json")


def _standalone_factory(store, folds, allow_cpu, cfg):
    lgbm = model_for(cfg, "lgbm", allow_cpu)

    def fn(cand) -> float:
        cs = ColSet((), tuple(cand.columns))
        run = run_config(store, lgbm, cs, folds, rounds=None, seed=cfg.sel_seed, keep_states=False)
        return float(np.median(run.gain_vs(run.e0)))

    return fn


def cmd_loop(cfg: RunConfig, args) -> None:
    """§2.1 cho model m từ B0*: calibrate riêng → add-one 39 candidate → prune PI → 3 seed → win_m → latency → §3 champion + figure."""
    gate(cfg, args, [args.model])
    store, folds, _, _ = load_store(cfg)
    mname = args.model
    exp = cfg.exp_dir
    is_probe = mname in PROBE_MODELS
    # Điểm xuất phát S: mọi model từ B0*, riêng nhánh `tfm_ext` từ ∅ (baseline = TimesFM native trên r1)
    base = ColSet((), ()) if TFM_BRANCH_BASE.get(mname) == "empty" else colset_from_arg(store, cfg, "b0star")
    model = model_for(cfg, args.model, args.allow_cpu)
    if load_champion(exp / "champion.json") is None and mname != "lgbm":
        sys.exit("§3: champion ban đầu phải là LightGBM code gốc — chạy `loop --model lgbm` trước.")
    # phase B calibrate riêng trên B0*
    base_label = "∅ (TimesFM native trên r1)" if not base.names else "B0*"
    say(f"[{mname}] calibrate trên {base_label} ({len(base.names)} cột) — ES với calib_seed {cfg.calib_seed}")
    cal = calibrate(store, model, base, folds, seed=cfg.calib_seed, keep_states=False)
    rounds = rounds_from(cal) if getattr(model, "supports_rounds", True) else None
    eps, noise, runs = seed_noise(store, model, base, folds, rounds, cfg.eval_seeds, cfg.eps_floor_pp)
    base_run = run_at_seed(runs, cfg.sel_seed)  # mốc của vòng lặp: cùng selection_seed với mọi candidate
    if base_run is None:
        base_run = run_config(store, model, base, folds, rounds=rounds, seed=cfg.sel_seed, keep_states=False)
    say(f"[{mname}] rounds={rounds} ε={eps:.4f} pp; base MedianGain vs E0 = {np.median(base_run.gain_vs(base_run.e0)):+.4f}")
    (exp / "calib").mkdir(parents=True, exist_ok=True)
    (exp / "calib" / f"{mname}_base.json").write_text(json.dumps(
        {"model": mname, "rounds": rounds, "eps": eps, "noise_cells": np.round(noise, 5).tolist(), "calib_seed": cfg.calib_seed,
         "eval_seeds": list(cfg.eval_seeds), "selection_seed": cfg.sel_seed, "rmse": base_run.rmse.tolist(),
         "e0": base_run.e0.tolist(), "seed_rmse": [r.rmse.tolist() for r in runs], "colset": base.to_dict()}, indent=1), encoding="utf-8")
    # add-one loop
    cands = candidates_from(cfg, args.max_candidates)
    standalone_fn = None if args.no_standalone else _standalone_factory(store, folds, args.allow_cpu, cfg)
    kd_path = exp / f"keepdrop_{mname}.csv"

    def on_row(row, run):
        exp_id = new_exp_id("loop", mname, row["candidate"])
        row["exp_id"] = exp_id  # vào keepdrop_<m>.csv (§7.2)
        say(f"[{mname}] {row['order']:02d} {row['candidate']:<28} Median {row['MedianGain_vs_S']:+.4f} → {row['decision']} (|S|={row['size_S_after']})")
        _log(cfg, exp_id=exp_id, step="loop", model=mname, seed=cfg.sel_seed, colset=row["columns"], n_cols=len(run.colset.names),
             rounds=json.dumps(rounds), base="S_m", MedianGain=row["MedianGain_vs_S"], WinRate=row["WinRate"], P10Gain=row["P10Gain"],
             WorstGain=row["WorstGain"], rmse_cells=row["rmse_cells"], mae_cells=_cells(run.mae), e0_cells=_cells(run.e0),
             gain_cells=row["gain_cells_vs_S"], decision=row["decision"], train_device=getattr(model, "train_device", ""))
        save_run(exp, exp_id, {**run.to_dict(), "step": "loop", "candidate": row["candidate"], "decision": row["decision"], "eps": eps,
                               "MedianGain_vs_S": row["MedianGain_vs_S"]})

    lr = add_one_loop(store, model, base, base_run.rmse, cands, folds, rounds, eps, cfg.sel_seed, base_run.e0, standalone_fn, on_row)
    lr.table.to_csv(kd_path, index=False)
    say(f"[{mname}] F*_m: {len(lr.kept)} KEEP / {len(lr.dropped)} DROP → {kd_path}")
    # dòng tổng kết: F*_m so với chính baseline của model (cùng selection_seed, cùng số vòng) — bắt buộc cho TimesFM (§2.2 #4b)
    g_fin = summarize(gain_pp(lr.final_rmse, base_run.rmse))
    _log(cfg, exp_id=new_exp_id("loop_final", mname), step="loop_final", model=mname, seed=cfg.sel_seed,
         colset="|".join(lr.final.ext), n_cols=len(lr.final.names), rounds=json.dumps(rounds), base="baseline_model",
         MedianGain=round(g_fin["MedianGain"], 4), WinRate=round(g_fin["WinRate"], 4), P10Gain=round(g_fin["P10Gain"], 4),
         WorstGain=round(g_fin["WorstGain"], 4), rmse_cells=_cells(lr.final_rmse), decision=f"F*_{mname}",
         note=f"{len(lr.kept)} KEEP / {len(lr.dropped)} DROP", train_device=getattr(model, "train_device", ""))
    say(f"[{mname}] F*_m vs baseline: MedianGain = {g_fin['MedianGain']:+.4f} pp")
    # prune PI
    pruned, pi_df = prune_pi(store, model, lr.final, folds, rounds, cfg.sel_seed)
    pi_df.to_csv(exp / f"prune_pi_{mname}.csv", index=False)
    say(f"[{mname}] prune PI: giữ {len(pruned.ext)}/{len(lr.final.ext)} cột ext")
    # confirmation 3 seed → win
    unp = confirm(store, model, lr.final, folds, cfg.eval_seeds, keep_states=True)
    prn = confirm(store, model, pruned, folds, cfg.eval_seeds, keep_states=True) if pruned.ext != lr.final.ext else unp
    which, g, s = decide_win(unp, prn, eps)
    win = prn if which == "prune" else unp
    for tag, conf in (("unprune", unp), ("prune", prn)):
        if tag == "prune" and prn is unp:
            continue
        for k, r in enumerate(conf.runs):
            eid = new_exp_id("confirm", mname, f"{tag}_seed{k}")
            _log(cfg, exp_id=eid, step="confirm", model=mname, seed=r.seed, colset=tag, rounds="ES", **_summ_row(r, r.e0, "E0"),
                 decision="", note=f"n_ext={len(conf.colset.ext)}", train_device=getattr(model, "train_device", ""))
            save_run(exp, eid, {**r.to_dict(), "step": "confirm", "configuration": tag, "eps": eps}, r.preds())
    pd.DataFrame([
        {"configuration": "unprune", "n_ext": len(lr.final.ext), "rmse_mean": json.dumps(np.round(unp.rmse_mean, 4).tolist())},
        {"configuration": "prune", "n_ext": len(pruned.ext), "rmse_mean": json.dumps(np.round(prn.rmse_mean, 4).tolist()),
         "MedianGain_prune_vs_unprune": s["MedianGain"], "WinRate": s["WinRate"], "P10Gain": s["P10Gain"], "WorstGain": s["WorstGain"],
         "eps": eps, "win": which},
    ]).to_csv(exp / f"prune_{mname}.csv", index=False)
    say(f"[{mname}] win_m = {which} (MedianGain prune vs unprune {s['MedianGain']:+.4f}, ε={eps:.4f})")
    win_dir = exp / "wins"
    win_dir.mkdir(parents=True, exist_ok=True)
    win_payload = {"model": mname, "colset": win.colset.to_dict(), "rmse_mean": win.rmse_mean.tolist(), "e0": win.e0.tolist(), "eps": eps,
                   "best_iters_by_seed": [b.tolist() for b in win.best_iters],
                   # seed THỰC SỰ đã chạy ở confirmation (model tất định như TimesFM chỉ có 1) — phải khớp số file *_seedK.npz
                   "eval_seeds": [int(r.seed) for r in win.runs], "which": which,
                   "folds": [f.name for f in folds],
                   "median_gain_vs_e0": float(np.median(gain_pp(win.rmse_mean, win.e0)))}
    (win_dir / f"{mname}.json").write_text(json.dumps(win_payload, indent=1), encoding="utf-8")
    for k, r in enumerate(win.runs):
        np.savez_compressed(win_dir / f"{mname}_seed{k}.npz", **{f"idx_{i}": p[0] for i, p in enumerate(r.preds())}, **{f"yhat_{i}": p[1] for i, p in enumerate(r.preds())})
    # latency (§7.4, chỉ theo dõi)
    lat = None
    try:
        lat = measure_tabular(win.runs[0], warmup=50, max_origins=args.latency_origins, model=model)
        lat.to_csv(exp / f"latency_{mname}.csv", index=False)
        log_latency(exp, lat, split="VAL")
        say(f"[{mname}] latency p95 (ms) per h: {lat['p95_ms'].round(3).tolist()} (predict device {lat['predict_device'].iloc[0]})")
    except Exception as e:  # không được ảnh hưởng pipeline
        say(f"[{mname}] latency bỏ qua: {e}")
    # §3 champion — model probe (AutoTS WR/MR) KHÔNG tham gia: chúng chỉ dò feature, AutoTS-final do `autots-search` sinh
    if is_probe:
        log_champion(exp, {"exp_id": new_exp_id("probe", mname), "model": mname, "win": which, "n_ext": len(win.colset.ext),
                           "ext_cols": "|".join(win.colset.ext), "MedianGain_vs_E0": round(win_payload["median_gain_vs_e0"], 4),
                           "rmse_mean_win": _cells(win.rmse_mean), "decision": "probe — không so champion",
                           "champion_after": (load_champion(exp / "champion.json") or {}).get("model", ""),
                           "train_device": getattr(model, "train_device", "")})
        say(f"[{mname}] probe feature-search xong ({len(win.colset.ext)} cột ext) — không so champion; "
            f"chạy `{FINAL_STEP[mname]}` để có model đại diện")
        return
    champ_path = exp / "champion.json"
    champ = load_champion(champ_path)
    win_mae = np.mean([r.mae for r in win.runs], axis=0)
    state = {"model": mname, "colset": win.colset.to_dict(), "rmse_mean": win.rmse_mean.tolist(), "mae_mean": win_mae.tolist(), "eps": eps,
             "e0": win.e0.tolist(), "which": which}
    exp_id = new_exp_id("champion", mname)
    row = {"exp_id": exp_id, "model": mname, "win": which, "n_ext": len(win.colset.ext), "ext_cols": "|".join(win.colset.ext),
           "MedianGain_vs_E0": round(win_payload["median_gain_vs_e0"], 4), "rmse_mean_win": _cells(win.rmse_mean),
           **{f"rmse_h{h}": round(float(win.rmse_mean[:, h - 1].mean()), 4) for h in HORIZONS},
           **{f"mae_h{h}": round(float(win_mae[:, h - 1].mean()), 4) for h in HORIZONS},
           "train_device": getattr(model, "train_device", ""), "predict_device": getattr(model, "predict_device", "")}
    if lat is not None:
        row.update({f"latency_{k}_ms": json.dumps(lat[f"{k}_ms"].round(3).tolist()) for k in ("p95", "p99", "max")})
    if champ is None:
        decision = "champion ban đầu"
        save_champion(champ_path, state)
        row.update({"champion_before": "", "decision": decision, "champion_after": mname})
        champ_tab_e0 = gain_pp(win.rmse_mean, win.e0)
        champ_label = mname
    else:
        change, gc, sc = compare(win.rmse_mean, np.asarray(champ["rmse_mean"]), float(champ["eps"]))
        decision = "đổi" if change else "giữ"
        ch_rmse = np.asarray(champ["rmse_mean"])
        row.update({"champion_before": champ["model"], "MedianGain_vs_champion": round(sc["MedianGain"], 4), "WinRate": round(sc["WinRate"], 4),
                    "P10Gain": round(sc["P10Gain"], 4), "WorstGain": round(sc["WorstGain"], 4), "eps_champion": champ["eps"], "decision": decision,
                    "rmse_mean_champion": _cells(ch_rmse), "gain_cells": _cells(gc),
                    **{f"champ_rmse_h{h}": round(float(ch_rmse[:, h - 1].mean()), 4) for h in HORIZONS}})
        champ_tab_e0 = gain_pp(ch_rmse, np.asarray(champ["e0"]))
        champ_label = champ["model"]
        if change:
            save_champion(champ_path, state)
        row["champion_after"] = mname if change else champ["model"]
    log_champion(exp, row)
    say(f"[{mname}] champion: {decision} (champion sau = {row['champion_after']})")
    # figure §7.3: win vs champion (forecast path của 3 origin đại diện)
    try:
        picks = plots.select_vol_origins(store, folds)
        win_preds = win.runs[0].preds()
        champ_preds = load_preds(win_dir / f"{champ_label}_seed0.npz") if (win_dir / f"{champ_label}_seed0.npz").exists() else win_preds
        series = [(f"win = {LABEL.get(mname, mname)}", win_preds, plots.WIN_STYLE[0], plots.WIN_STYLE[1]),
                  (f"champion = {LABEL.get(champ_label, champ_label)}", champ_preds, plots.CHAMP_STYLE[0], plots.CHAMP_STYLE[1])]
        plots.fig_path(store, picks, series, exp / "summary" / f"fig_path_{mname}_vs_champion.png",
                       f"Fig P — forecast path win vs champion ({mname} vs {champ_label}): x = t → t+3, y = thay đổi giá so với C_t")
        for h in HORIZONS:  # Fig T: trajectory toàn bộ VAL (5 fold, không nối qua ranh giới fold)
            plots.fig_trajectory(store, h, series, exp / "summary" / f"fig_traj_h{h}_{mname}_vs_champion.png",
                                 f"Fig T{h} — trajectory VAL ({mname} vs {champ_label}): actual C_(t+{h}) vs P̂_(t+{h}) = C_t·exp(ŷ_{h})")
        footer = f"win vs champion: {decision}" + ("" if champ is None else f" — MedianGain {row['MedianGain_vs_champion']:+.4f} (ε {champ['eps']:.4f})")
        plots.fig_hm(gain_pp(win.rmse_mean, win.e0), champ_tab_e0, [f.name.split('_')[-1] for f in folds], LABEL.get(mname, mname),
                     LABEL.get(champ_label, champ_label), footer, exp / "summary" / f"fig_HM_{mname}_vs_champion.png")
    except Exception as e:
        say(f"[{mname}] figure bỏ qua: {e}")


DEFAULT_AUTOTS_TEMPLATES = [  # bake-off phương án A: mọi dòng GPU; nhóm theo shift regressor (wr:<window> / mr)
    {"model": "wr", "window_size": 60, "regressor": "LightGBM"},
    {"model": "wr", "window_size": 60, "regressor": "xgboost"},
    {"model": "mr", "regressor": "xgboost"},
    {"model": "mr", "regressor": "LightGBM"},
]


def _autots_group(spec: dict) -> str:
    """Các dòng template dùng CHUNG một `future_regressor` nên phải cùng phép dịch: MR = f(s−1), WR = f(s+W−1).
    → chia bake-off theo nhóm cùng shift; chọn giữa các nhóm bằng metric của project trên outer VAL (không phải điểm AutoTS)."""
    kind = str(spec.get("model", "wr")).lower()
    kind = {"windowregression": "wr", "multivariateregression": "mr"}.get(kind, kind)
    return "mr" if kind == "mr" else f"wr:{int(spec.get('window_size', 60))}"


def _autots_probe_model(cfg: RunConfig, group: str, allow_cpu: bool, frozen=None):
    from .models_autots import AutoTSModel

    params = _params_for(cfg, "autots_wr" if group.startswith("wr") else "autots_mr")
    params.pop("window_size", None)
    kw = dict(kind="mr" if group == "mr" else "wr", allow_cpu=allow_cpu, frozen=frozen, **params)
    if group.startswith("wr:"):
        kw["window_size"] = int(group.split(":")[1])
    if allow_cpu:
        kw["device"] = "cpu"
    return AutoTSModel(**kw)


def autots_bakeoff_fold(cfg: RunConfig, store: Store, fold, colset: ColSet, group: str, specs: list[dict],
                        nv: int, allow_cpu: bool, cov_all=None) -> tuple[str, dict, object]:
    """Bake-off template GPU trên TRAINING-SIDE của một fold (FIT+ES, dừng trước purge) → (tên model, params, bảng candidate).

    Dùng chung cho `autots-search` (5 fold VAL) và cho `final` (fold final, để refit AutoTS-final trên TEST).
    Outer VAL/TEST không bao giờ nằm trong `df_tr`.
    """
    from .autots_search import search_best_template, template_frame
    from .harness import _standardize_fit
    from .models import SeriesBatch

    idx_fit = fold.fit.origins(store.ts, store.eligible)
    idx_es = fold.es.origins(store.ts, store.eligible)
    cov = _standardize_fit(store.grid_matrix(colset) if cov_all is None else cov_all, idx_fit)
    seq = SeriesBatch(store.ts, store.r1, np.concatenate([idx_fit, idx_es]), cov, tuple(colset.names))
    lo, hi = int(idx_fit.min()), int(idx_es.max()) + 1
    probe = _autots_probe_model(cfg, group, allow_cpu)
    df_tr, R_tr = probe.frames(seq, lo, hi)
    say(f"[{fold.name}|{group}] search trên {len(df_tr)} bar (đến {df_tr.index[-1]}), {len(specs)} template × {nv} validation")
    name, params, all_t = search_best_template(df_tr, R_tr, template_frame(specs, seed=cfg.sel_seed), nv, cfg.sel_seed)
    return name, params, all_t


def autots_search_cfg(cfg: RunConfig) -> tuple[dict, int]:
    """(nhóm shift → danh sách template, num_validations) từ config; mặc định = DEFAULT_AUTOTS_TEMPLATES."""
    c = cfg.model_params("autots_search")
    specs = c.get("templates") or DEFAULT_AUTOTS_TEMPLATES
    groups = {}
    for sp in specs:
        groups.setdefault(_autots_group(sp), []).append(sp)
    return groups, int(c.get("num_validations", 10))


def cmd_autots_search(cfg: RunConfig, args) -> None:
    """§2.2 #6 (iii) — chạy framework AutoTS trên HAI feature set đã freeze → AutoTS-final.

    Input = `wins/autots_wr.json` và `wins/autots_mr.json` (feature set sau prune + confirmation của từng probe).
    Vai trò seed (§1.3): template search + **chọn candidate** chạy ở `selection_seed`; sau khi AutoTS-final đã FREEZE
    (feature set + template từng fold) mới chạy lại trên 3 `eval_seeds` để lấy RMSE̅, noise, **ε của chính AutoTS-final**
    và prediction seed0/1/2 cho ensemble.

    Với mỗi frozen set (F_WR_best / F_MR_best; dedup nếu trùng) và mỗi nhóm shift: mỗi fold chạy `AutoTS` với
    `initial_template` do ta khai báo (mọi dòng GPU) + `max_generations=0` **chỉ trên training-side (FIT+ES,
    kết thúc trước purge 60')** → template thắng được FREEZE → refit + rolling predict outer VAL bằng ModelMonster.
    Mọi so sánh cắt ngang nhóm/feature set đều dùng **metric của project** (RMSE/Gain 15 ô → MedianGain),
    KHÔNG dùng điểm nội bộ của AutoTS. Feature set không đổi trong suốt bake-off (assert trong `search_best_template`).
    """
    gate(cfg, args, ["autots_wr", "autots_mr"])
    exp = cfg.exp_dir
    frozen_sets = {}
    for m in ("autots_wr", "autots_mr"):  # kiểm tra TRƯỚC khi đọc data: framework chỉ chạy sau khi feature set đã freeze
        p = exp / "wins" / f"{m}.json"
        if not p.exists():
            sys.exit(f"Thiếu {p} — phải chạy `loop --model {m}` (add-one → prune → confirmation) trước; "
                     "win sau confirmation chính là feature set được freeze.")
        w = json.loads(p.read_text(encoding="utf-8"))
        frozen_sets[f"F_{m.split('_')[1].upper()}_best"] = ColSet.from_dict(w["colset"])
    store, folds, _, _ = load_store(cfg)
    names = list(frozen_sets)
    if frozen_sets[names[0]].names == frozen_sets[names[1]].names:
        say(f"{names[0]} và {names[1]} trùng nhau → dedup, chỉ chạy framework một lần")
        frozen_sets = {f"{names[0]}={names[1]}": frozen_sets[names[0]]}
    groups, nv = autots_search_cfg(cfg)
    say(f"bake-off: {sum(len(v) for v in groups.values())} template / {len(groups)} nhóm shift {list(groups)}, "
        f"num_validations={nv}, {len(frozen_sets)} frozen set × {len(folds)} fold")
    tmpl_dir = exp / "autots_templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    rows, cands = [], {}
    for set_name, colset in frozen_sets.items():
        cov_all = store.grid_matrix(colset)
        for group, gspecs in groups.items():
            frozen_by_fold = {}
            for f in folds:
                name, params, all_t = autots_bakeoff_fold(cfg, store, f, colset, group, gspecs, nv, args.allow_cpu, cov_all)
                frozen_by_fold[f.name] = (name, params)
                tag = f"{set_name}_{group.replace(':', '')}_{f.name}".replace("=", "_")
                (tmpl_dir / f"best_{tag}.json").write_text(json.dumps({"set": set_name, "group": group, "fold": f.name,
                                                                      "model": name, "params": params}, indent=1), encoding="utf-8")
                try:
                    all_t.to_json(tmpl_dir / f"all_{tag}.json", orient="records", indent=1)
                except Exception:
                    pass
                say(f"[{set_name}|{group}|{f.name}] template thắng: {name}")
            # CHỌN candidate: chấm outer VAL ở ĐÚNG `selection_seed` (§1.3 — không dùng mean 3 eval seed để chọn)
            fold_rmse, e0_rows = [], []
            for f in folds:
                m = _autots_probe_model(cfg, group, args.allow_cpu, frozen=frozen_by_fold[f.name])
                r = run_config(store, m, colset, [f], rounds=None, seed=cfg.sel_seed, keep_states=False)
                fold_rmse.append(r.rmse[0])
                e0_rows.append(r.e0[0])  # E0 theo TỪNG fold (không phụ thuộc seed)
            rmse_sel, e0_tab = np.array(fold_rmse), np.array(e0_rows)
            key = f"{set_name}|{group}"
            cands[key] = {"set": set_name, "group": group, "colset": colset, "rmse_sel": rmse_sel, "e0": e0_tab,
                          "templates": frozen_by_fold}
            g = float(np.median(gain_pp(rmse_sel, e0_tab)))
            rows.append({"candidate": key, "set": set_name, "group": group, "n_ext": len(colset.ext),
                         "MedianGain_vs_E0_sel": round(g, 4), "rmse_selection_seed": _cells(rmse_sel),
                         "templates": "|".join(sorted({v[0] for v in frozen_by_fold.values()}))})
            say(f"[{key}] outer VAL @ selection_seed {cfg.sel_seed}: MedianGain vs E0 = {g:+.4f} pp")
    final_key = max(cands, key=lambda k: float(np.median(gain_pp(cands[k]["rmse_sel"], cands[k]["e0"]))))
    fin = cands[final_key]
    say(f"AutoTS-final = {final_key} ({len(fin['colset'].ext)} cột ext) — chọn ở selection_seed {cfg.sel_seed}")
    # CONFIRMATION: winner đã FREEZE (feature set + template từng fold) → chạy lại trên 3 evaluation seed
    tables, preds_by_seed = [], []
    for sd in cfg.eval_seeds:
        fold_rmse, preds = [], []
        for f in folds:
            m = _autots_probe_model(cfg, fin["group"], args.allow_cpu, frozen=fin["templates"][f.name])
            r = run_config(store, m, fin["colset"], [f], rounds=None, seed=sd, keep_states=True)
            fold_rmse.append(r.rmse[0])
            preds.append(r.preds()[0])
        tables.append(np.array(fold_rmse))
        preds_by_seed.append(preds)
    rmse_mean = mean_rmse_over_seeds(tables)
    noise = seed_noise_cells(tables)
    eps = seed_noise_eps(tables, cfg.eps_floor_pp)  # ε của CHÍNH AutoTS-final, không mượn ε của probe
    g_fin = float(np.median(gain_pp(rmse_mean, fin["e0"])))
    say(f"AutoTS-final confirmation {list(cfg.eval_seeds)}: MedianGain vs E0 = {g_fin:+.4f} pp, ε = {eps:.4f} pp")
    for r_ in rows:
        if r_["candidate"] == final_key:
            r_.update({"is_final": True, "MedianGain_vs_E0_confirm": round(g_fin, 4), "eps_autots_final": round(eps, 5)})
    pd.DataFrame(rows).to_csv(exp / "autots_search.csv", index=False)
    win_dir = exp / "wins"
    win_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": "autots", "role": "AutoTS-final", "source": final_key, "group": fin["group"],
               "colset": fin["colset"].to_dict(),
               "rmse_mean": rmse_mean.tolist(), "e0": fin["e0"].tolist(), "eps": eps,
               "noise_cells": np.round(noise, 5).tolist(), "seed_rmse": [t.tolist() for t in tables],
               "rmse_selection_seed": fin["rmse_sel"].tolist(), "selection_seed": cfg.sel_seed,
               "eval_seeds": [int(sd) for sd in cfg.eval_seeds], "folds": [f.name for f in folds],
               "templates_per_fold": {k: v[0] for k, v in fin["templates"].items()},
               "median_gain_vs_e0": g_fin}
    (win_dir / "autots.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, preds in enumerate(preds_by_seed):
        np.savez_compressed(win_dir / f"autots_seed{k}.npz", **{f"idx_{i}": p[0] for i, p in enumerate(preds)},
                            **{f"yhat_{i}": p[1] for i, p in enumerate(preds)})
    _log(cfg, exp_id=new_exp_id("autots_search", "autots"), step="autots_search", model="autots", seed=cfg.sel_seed,
         colset="|".join(fin["colset"].ext), n_cols=len(fin["colset"].names), rounds="bake-off template",
         base="E0", MedianGain=round(g_fin, 4), rmse_cells=_cells(rmse_mean), e0_cells=_cells(fin["e0"]),
         decision=f"AutoTS-final={final_key}",
         note=f"chọn @seed {cfg.sel_seed}; confirmation {list(cfg.eval_seeds)}; ε={eps:.4f}")
    champion_step(cfg, "autots", fin["colset"], rmse_mean, fin["e0"], eps, {"win": "autots_final"})


def champion_step(cfg: RunConfig, mname: str, colset: ColSet, rmse_mean: np.ndarray, e0: np.ndarray, eps: float,
                  extra: dict | None = None) -> str:
    """§3: so bảng RMSE̅ của win_m với champion → đổi/giữ, ghi champion_log.csv. Trả tên champion sau."""
    exp = cfg.exp_dir
    champ_path = exp / "champion.json"
    champ = load_champion(champ_path)
    if champ is None and mname != "lgbm":  # §3: champion ban đầu phải là LightGBM code gốc
        sys.exit(f"§3: chưa có champion — chạy `loop --model lgbm` trước khi để '{mname}' so champion.")
    state = {"model": mname, "colset": colset.to_dict(), "rmse_mean": rmse_mean.tolist(), "eps": eps, "e0": e0.tolist()}
    row = {"exp_id": new_exp_id("champion", mname), "model": mname, "n_ext": len(colset.ext), "ext_cols": "|".join(colset.ext),
           "MedianGain_vs_E0": round(float(np.median(gain_pp(rmse_mean, e0))), 4), "rmse_mean_win": _cells(rmse_mean),
           **{f"rmse_h{h}": round(float(rmse_mean[:, h - 1].mean()), 4) for h in HORIZONS}, **(extra or {})}
    if champ is None:
        save_champion(champ_path, state)
        row.update({"champion_before": "", "decision": "champion ban đầu", "champion_after": mname})
    else:
        change, gc, sc = compare(rmse_mean, np.asarray(champ["rmse_mean"]), float(champ["eps"]))
        row.update({"champion_before": champ["model"], "MedianGain_vs_champion": round(sc["MedianGain"], 4),
                    "WinRate": round(sc["WinRate"], 4), "P10Gain": round(sc["P10Gain"], 4), "WorstGain": round(sc["WorstGain"], 4),
                    "eps_champion": champ["eps"], "decision": "đổi" if change else "giữ", "gain_cells": _cells(gc),
                    "rmse_mean_champion": _cells(np.asarray(champ["rmse_mean"])),
                    "champion_after": mname if change else champ["model"]})
        if change:
            save_champion(champ_path, state)
    log_champion(exp, row)
    say(f"[{mname}] champion: {row['decision']} (champion sau = {row['champion_after']})")
    return str(row["champion_after"])


def cmd_tfm_final(cfg: RunConfig, args) -> None:
    """§2.2 #4 — chọn TimesFM-final giữa hai nhánh đã hoàn tất (mỗi nhánh đã qua add-one → prune → confirmation).

    Không chạy lại model: hai nhánh đã có bảng `RMSE̅` và prediction của chính chúng; ở đây chỉ so bằng metric
    của project (MedianGain vs E0 trên 15 ô) rồi ghi model đại diện `tfm` để đi champion/ensemble/Final.
    """
    exp = cfg.exp_dir
    wins = {}
    for m in ("tfm_b0", "tfm_ext"):
        p = exp / "wins" / f"{m}.json"
        if not p.exists():
            sys.exit(f"Thiếu {p} — phải chạy `loop --model {m}` cho CẢ HAI nhánh TimesFM trước.")
        wins[m] = json.loads(p.read_text(encoding="utf-8"))
    gate(cfg, args, [])
    rows = []
    for m, w in wins.items():
        g = float(np.median(gain_pp(np.asarray(w["rmse_mean"]), np.asarray(w["e0"]))))
        rows.append({"branch": m, "n_ext": len(w["colset"]["ext"]), "n_b0": len(w["colset"]["b0"]),
                     "ext_cols": "|".join(w["colset"]["ext"]), "which": w.get("which", ""),
                     "MedianGain_vs_E0": round(g, 4), "rmse_mean": _cells(np.asarray(w["rmse_mean"]))})
        say(f"[{m}] MedianGain vs E0 = {g:+.4f} pp ({len(w['colset']['b0'])} cột B0*, {len(w['colset']['ext'])} cột ext)")
    pd.DataFrame(rows).to_csv(exp / "tfm_final.csv", index=False)
    best = max(rows, key=lambda r: r["MedianGain_vs_E0"])["branch"]
    w = wins[best]
    say(f"TimesFM-final = {best} (chọn bằng metric project trên VAL)")
    payload = {**w, "model": "tfm", "role": "TimesFM-final", "branch": best,
               "covariate_scope": "b0star" if best == "tfm_b0" else "ext"}
    (exp / "wins" / "tfm.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in range(len(w.get("eval_seeds", cfg.eval_seeds))):
        src = exp / "wins" / f"{best}_seed{k}.npz"
        if src.exists():
            (exp / "wins" / f"tfm_seed{k}.npz").write_bytes(src.read_bytes())
    _log(cfg, exp_id=new_exp_id("tfm_final", "tfm"), step="tfm_final", model="tfm", seed=cfg.sel_seed,
         colset="|".join(w["colset"]["ext"]), n_cols=len(w["colset"]["b0"]) + len(w["colset"]["ext"]),
         rounds="zero-shot", base="E0", MedianGain=round(float(np.median(gain_pp(np.asarray(w["rmse_mean"]),
                                                                                np.asarray(w["e0"])))), 4),
         rmse_cells=_cells(np.asarray(w["rmse_mean"])), e0_cells=_cells(np.asarray(w["e0"])),
         decision=f"TimesFM-final={best}", note="chọn giữa nhánh B0* và nhánh ext")
    champion_step(cfg, "tfm", ColSet.from_dict(w["colset"]), np.asarray(w["rmse_mean"]), np.asarray(w["e0"]),
                  float(w["eps"]), {"win": f"tfm_final={best}"})


def cmd_ensemble(cfg: RunConfig, args) -> None:
    """§3 ensemble: thành viên = champion + mọi win_m có MedianGain vs E0 > 0; (a) đều, (b) 1/MSE; so với champion."""
    store, folds, _, _ = load_store(cfg)
    exp = cfg.exp_dir
    champ = load_champion(exp / "champion.json")
    if champ is None:
        sys.exit("Chưa có champion.")
    wins = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (exp / "wins").glob("*.json")
            if p.stem not in PROBE_MODELS}  # probe WR/MR không phải thành viên; AutoTS-final (nếu có) mới là
    members = [m for m, w in wins.items() if w["median_gain_vs_e0"] > 0 or m == champ["model"]]
    if len(members) < 2:
        say("Không đủ 2 thành viên — không ensemble.")
        return
    preds = {m: [load_preds(exp / "wins" / f"{m}_seed{k}.npz") for k in range(len(wins[m]["eval_seeds"]))] for m in members}
    eq = ensemble_rmse(store, preds, folds)
    inv = ensemble_rmse(store, preds, folds, inverse_mse_weights({m: np.asarray(wins[m]["rmse_mean"]) for m in members}))
    champ_tab = np.asarray(champ["rmse_mean"])
    res = {}
    for name, tab in (("equal", eq), ("inv_mse", inv)):
        change, g, s = compare(tab, champ_tab, float(champ["eps"]))
        res[name] = (change, s, tab)
        say(f"ensemble {name}: MedianGain vs champion {s['MedianGain']:+.4f} → {'đổi' if change else 'giữ'}")
    best = max(res, key=lambda k: res[k][1]["MedianGain"])
    change, s, tab = res[best]
    g_best = gain_pp(tab, champ_tab)
    row = {"exp_id": new_exp_id("ensemble", best), "model": f"ensemble_{best}", "members": "|".join(members), "weighting": best,
           "MedianGain_vs_E0": round(float(np.median(gain_pp(tab, np.asarray(champ["e0"])))), 4), "champion_before": champ["model"],
           "MedianGain_vs_champion": round(s["MedianGain"], 4), "WinRate": round(s["WinRate"], 4), "P10Gain": round(s["P10Gain"], 4),
           "WorstGain": round(s["WorstGain"], 4), "eps_champion": champ["eps"], "decision": "đổi" if change else "giữ",
           "champion_after": f"ensemble_{best}" if change else champ["model"], "rmse_mean_win": _cells(tab), "rmse_mean_champion": _cells(champ_tab),
           "gain_cells": _cells(g_best), **{f"rmse_h{h}": round(float(tab[:, h - 1].mean()), 4) for h in HORIZONS},
           **{f"champ_rmse_h{h}": round(float(champ_tab[:, h - 1].mean()), 4) for h in HORIZONS}}
    log_champion(exp, row)
    (exp / "ensemble.json").write_text(json.dumps({"members": members, "weighting": best, "rmse_mean": tab.tolist(), "is_champion": change}, indent=1), encoding="utf-8")
    if change:
        save_champion(exp / "champion.json", {"model": f"ensemble_{best}", "members": members, "weighting": best, "rmse_mean": tab.tolist(),
                                              "eps": champ["eps"], "e0": champ["e0"]})


def cmd_final(cfg: RunConfig, args) -> None:
    """§4 Final một lần: refit B0-306, B0*, mọi win_m (+ ensemble) trên fold final → TEST; all_models.csv; heatmap + Fig H_h mọi model."""
    gate(cfg, args, [p.stem for p in (cfg.exp_dir / "wins").glob("*.json")] or ["lgbm"])
    store, folds, final, _ = load_store(cfg)
    exp = cfg.exp_dir
    configs: dict[str, tuple[str, ColSet]] = {"b0_306": ("lgbm", store.all_b0())}
    if (exp / "b0_star.json").exists():
        configs["b0_star"] = ("lgbm", ColSet.load(exp / "b0_star.json"))
    for p in sorted((exp / "wins").glob("*.json")):
        if p.stem in PROBE_MODELS:  # probe WR/MR chỉ dò feature — AutoTS trên TEST là AutoTS-final (`autots`)
            continue
        w = json.loads(p.read_text(encoding="utf-8"))
        configs[w["model"]] = (w["model"], ColSet.from_dict(w["colset"]))
    idx_test = final.val.origins(store.ts, store.eligible)
    c_t, c_future, _ = store.targets(idx_test)
    block = (store.ts[idx_test] - final.val.start) // (6 * 3600)
    blocks = sorted(set(block.tolist()))
    block_labels = [f"{pd.Timestamp(final.val.start + b * 21600, unit='s', tz='UTC').strftime('%m-%d %H')}h" for b in blocks]
    rows, tables, preds_by_model, ref_rmse = [], {}, {}, {}
    (exp / "summary").mkdir(parents=True, exist_ok=True)
    e0 = e0_rmse(c_t, c_future)
    e0_blocks = np.array([e0_rmse(c_t[block == b], c_future[block == b]) for b in blocks])
    rows.append({"model": "e0", **{f"rmse_h{h}": e0[h - 1] for h in HORIZONS}})
    yhat_by_model = {}

    def add_row(key, m, yhat, extra):
        ref_rmse[key] = m["rmse"]
        rows.append({"model": key, **{f"rmse_h{h}": m["rmse"][h - 1] for h in HORIZONS}, **{f"mae_h{h}": m["mae"][h - 1] for h in HORIZONS},
                     **{f"r_h{h}": m["r"][h - 1] for h in HORIZONS}, **{f"diracc_h{h}": m["dir_acc"][h - 1] for h in HORIZONS},
                     **{f"gain_e0_h{h}": gain_pp(m["rmse"], e0)[h - 1] for h in HORIZONS},
                     **{f"gain_b0306_h{h}": gain_pp(m["rmse"], ref_rmse["b0_306"])[h - 1] for h in HORIZONS if "b0_306" in ref_rmse},
                     **{f"gain_b0star_h{h}": gain_pp(m["rmse"], ref_rmse["b0_star"])[h - 1] for h in HORIZONS if "b0_star" in ref_rmse}, **extra})
        tables[key] = np.array([gain_pp(cell_metrics(c_t[block == b], c_future[block == b], yhat[block == b])["rmse"], e0_blocks[i]) for i, b in enumerate(blocks)])
        preds_by_model[key] = [(idx_test, yhat)]

    for key, (mname, cs) in configs.items():
        if mname == "tfm":  # TimesFM-final: dùng đúng scope covariate của nhánh đã thắng
            w = json.loads((exp / "wins" / "tfm.json").read_text(encoding="utf-8"))
            model = make_model("tfm", {**_params_for(cfg, "tfm"), "covariate_scope": w.get("covariate_scope", "ext")},
                               allow_cpu=args.allow_cpu)
        elif mname == "autots":  # AutoTS-final: bake-off lại trên training-side của fold final rồi freeze template
            w = json.loads((exp / "wins" / "autots.json").read_text(encoding="utf-8"))
            groups, nv = autots_search_cfg(cfg)
            grp = w.get("group", next(iter(groups)))
            name, params, _ = autots_bakeoff_fold(cfg, store, final, cs, grp, groups[grp], nv, args.allow_cpu)
            (exp / "autots_templates").mkdir(parents=True, exist_ok=True)
            (exp / "autots_templates" / "best_FINAL.json").write_text(json.dumps(
                {"fold": final.name, "group": grp, "model": name, "params": params}, indent=1), encoding="utf-8")
            model = _autots_probe_model(cfg, grp, args.allow_cpu, frozen=(name, params))
        else:
            model = model_for(cfg, mname, args.allow_cpu)
        run = run_config(store, model, cs, [final], rounds=None, seed=cfg.sel_seed, keep_states=True)
        yhat = run.states[0].yhat
        yhat_by_model[key] = yhat
        m = cell_metrics(c_t, c_future, yhat)
        extra = {"best_iters": json.dumps(run.best_iters.tolist()), "n_ext": len(cs.ext), "train_device": getattr(model, "train_device", ""),
                 "predict_device": getattr(model, "predict_device", "")}
        try:
            lat = measure_tabular(run, warmup=50, max_origins=args.latency_origins, model=model)
            lat["model"] = key  # b0_306 / b0_star / lgbm cùng model LightGBM → phân biệt theo cấu hình
            lat.to_csv(exp / "summary" / f"latency_final_{key}.csv", index=False)
            log_latency(exp, lat, split="TEST")
            for _, lr in lat.iterrows():
                extra.update({f"lat_{k}_h{int(lr['h'])}": round(float(lr[f"{k}_ms"]), 3) for k in ("p95", "p99", "max")})
        except Exception as e:
            say(f"latency {key} bỏ qua: {e}")
        add_row(key, m, yhat, extra)
        exp_id = new_exp_id("final", key)
        _log(cfg, exp_id=exp_id, step="final", model=mname, seed=cfg.sel_seed, colset=key, rounds="ES", **_summ_row(run, run.e0, "E0"),
             train_device=getattr(model, "train_device", ""), decision="TEST", note="final_TEST")
        save_run(exp, exp_id, {**run.to_dict(), "step": "final", "key": key, "test_metrics": {k: np.asarray(v).tolist() for k, v in m.items()}},
                 [(idx_test, yhat)], pred_name="pred_test.npz")
        say(f"final {key}: RMSE {np.round(m['rmse'], 2).tolist()} | Gain vs E0 {np.round(gain_pp(m['rmse'], e0), 4).tolist()}")
    ens_path = exp / "ensemble.json"
    if ens_path.exists():
        ens = json.loads(ens_path.read_text(encoding="utf-8"))
        mem = [m for m in ens["members"] if m in yhat_by_model]
        if len(mem) >= 2:
            w = {m: np.ones(3) for m in mem} if ens["weighting"] == "equal" else inverse_mse_weights({m: np.asarray(json.loads((exp / "wins" / f"{m}.json").read_text())["rmse_mean"]) for m in mem})
            acc = sum(yhat_by_model[m] * w[m][None, :] for m in mem)
            yhat = acc / sum(w[m] for m in mem)[None, :]
            yhat_by_model["ensemble"] = yhat
            add_row("ensemble", cell_metrics(c_t, c_future, yhat), yhat, {"members": "|".join(mem), "weighting": ens["weighting"]})
    champ = load_champion(exp / "champion.json")
    champ_key = None if champ is None else ("ensemble" if str(champ["model"]).startswith("ensemble") else str(champ["model"]))
    if champ_key in ref_rmse:
        for r in rows:
            if r["model"] in ref_rmse:
                r.update({f"gain_champion_h{h}": gain_pp(ref_rmse[r["model"]], ref_rmse[champ_key])[h - 1] for h in HORIZONS})
        say(f"champion trên TEST = {champ_key}")
    pd.DataFrame(rows).to_csv(exp / "summary" / "all_models_test.csv", index=False)
    plots.final_heatmaps(tables, block_labels, exp / "summary" / "fig_final_heatmaps.png")
    picks = plots.select_vol_origins_test(store, final.val)
    plots.final_fig_paths(store, picks, preds_by_model, exp / "summary" / "fig_final_paths_all_models.png")
    for h in HORIZONS:  # Fig T: trajectory toàn bộ TEST cho mọi model đang được vẽ
        plots.final_fig_trajectory(store, h, preds_by_model, exp / "summary" / f"fig_final_traj_h{h}_all_models.png")
    say(f"final → {exp / 'summary'}")


def cmd_smoke_e2e(cfg_unused, args) -> None:
    """Toàn bộ pipeline trên data tổng hợp, CPU (chỉ debug). Tạo cfg tạm trong --out."""
    from .synthetic import make_hf, make_lf

    out = Path(args.out).resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    hf = make_hf(n_days=args.days, seed=1)
    lf = make_lf(hf)
    hf.to_csv(out / "data" / "hf.csv", index=False)
    lf.to_csv(out / "data" / "lf.csv", index=False)
    start = pd.Timestamp(hf["timestamp"].iloc[0], unit="s", tz="UTC")
    days = [(start + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, int(args.days) - 1)]
    val_days = days[-3:]
    test_day = (start + pd.Timedelta(days=int(args.days) - 1)).strftime("%Y-%m-%d 00:00:00")
    cfg = RunConfig(dataset_label="synthetic_smoke", hf_csv="data/hf.csv", lf_csv="data/lf.csv", val_days=val_days, test_start=test_day,
                    calib_seed=1, eval_seeds=(2, 3, 4), selection_seed=2, experiments_dir="experiments", require_gpu=False,
                    root=str(out), candidates=[c.name for c in CANDIDATES])
    (out / "configs").mkdir(exist_ok=True)
    (out / "configs" / "smoke.json").write_text(json.dumps({k: v for k, v in cfg.to_dict().items() if k != "root"}, indent=1), encoding="utf-8")
    (out / ".claude").mkdir(exist_ok=True)
    (out / ".claude" / "MEMORY.md").write_text("TRAINING: UNLOCKED\n", encoding="utf-8")
    ns = argparse.Namespace(smoke=True, allow_cpu=True, write_checksums=True, model="lgbm", colset="b0306", max_cols=6, max_candidates=3,
                            no_standalone=False, latency_origins=100)
    cmd_check_data(cfg, ns)
    cmd_calibrate(cfg, ns)
    cmd_filter_b0(cfg, ns)
    cmd_loop(cfg, ns)
    ns.model = "xgb"
    cmd_loop(cfg, ns)
    cmd_ensemble(cfg, ns)
    cmd_final(cfg, ns)
    say(f"smoke-e2e OK → {out / 'experiments'}")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    def common(top: bool) -> argparse.ArgumentParser:
        # cờ chung nhận ở cả hai vị trí: `run.py --config X step` và `run.py step --config X`; subparser dùng SUPPRESS để không ghi đè
        c = argparse.ArgumentParser(add_help=False)
        sup = argparse.SUPPRESS
        c.add_argument("--config", default="configs/p0_15d.json" if top else sup)
        c.add_argument("--smoke", action="store_true", default=False if top else sup,
                       help="bỏ qua khóa training + GPU gate — CHỈ chấp nhận với dataset_label 'synthetic*' (data tổng hợp / debug)")
        c.add_argument("--allow-cpu", action="store_true", default=False if top else sup,
                       help="ép model chạy CPU — CHỈ chấp nhận với dataset_label 'synthetic*' (unit/smoke test)")
        return c

    p = argparse.ArgumentParser(prog="p0", description="P0_forecasting harness (docs/RESEARCH_PLAN.md)", parents=[common(True)])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check-data", parents=[common(False)]); s.add_argument("--write-checksums", action="store_true")
    s = sub.add_parser("calibrate", parents=[common(False)]); s.add_argument("--model", default="lgbm"); s.add_argument("--colset", default="b0306")
    s = sub.add_parser("filter-b0", parents=[common(False)]); s.add_argument("--max-cols", type=int, default=None)
    s = sub.add_parser("loop", parents=[common(False)]); s.add_argument("--model", required=True); s.add_argument("--max-candidates", type=int, default=None)
    s.add_argument("--no-standalone", action="store_true"); s.add_argument("--latency-origins", type=int, default=None)
    sub.add_parser("tfm-final", parents=[common(False)])
    sub.add_parser("autots-search", parents=[common(False)])
    sub.add_parser("ensemble", parents=[common(False)])
    s = sub.add_parser("final", parents=[common(False)]); s.add_argument("--latency-origins", type=int, default=None)
    s = sub.add_parser("smoke-e2e", parents=[common(False)]); s.add_argument("--out", default="tmp_smoke"); s.add_argument("--days", type=float, default=6)
    args = p.parse_args(argv)
    if args.cmd == "smoke-e2e":
        cmd_smoke_e2e(None, args)
        return
    cfg = RunConfig.load(args.config)
    {"check-data": cmd_check_data, "calibrate": cmd_calibrate, "filter-b0": cmd_filter_b0, "loop": cmd_loop,
     "tfm-final": cmd_tfm_final, "autots-search": cmd_autots_search, "ensemble": cmd_ensemble,
     "final": cmd_final}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
