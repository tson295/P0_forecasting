"""CLI: python run.py <step> --config configs/p0_15d.json [--model lgbm] [--smoke] [--allow-cpu]

Bước (§8): check-data → calibrate (lgbm, b0306) → filter-b0 → loop --model m (calibrate riêng trên B0*, add-one, prune PI,
3 seed → win_m, latency, champion + figure) → ensemble → final. `smoke-e2e` chạy toàn bộ trên data tổng hợp CPU (chỉ debug).
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
from .harness import ColSet, Store, calibrate, rounds_from, run_config, seed_noise
from .latency import measure_tabular
from .logs import load_preds, log_champion, log_latency, log_run, new_exp_id, save_run
from .loop import (add_one_loop, compare, confirm, decide_win, ensemble_rmse, inverse_mse_weights, load_champion, prune_pi,
                   save_champion)
from .metrics import cell_metrics, e0_rmse, gain_pp, summarize
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
    elif model == "lstm":
        import torch

        if not torch.cuda.is_available():
            sys.exit("GPU preflight LSTM: CUDA không có — cấm training CPU.")
        say(f"GPU preflight torch: {torch.cuda.get_device_name(0)}")


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


def model_for(cfg: RunConfig, name: str, allow_cpu: bool):
    if allow_cpu and not _is_synthetic(cfg):
        sys.exit(f"allow_cpu với dataset '{cfg.dataset_label}' bị cấm (plan §0: training chỉ GPU).")
    params = cfg.model_params(name)
    if allow_cpu:  # smoke/unit: ép CPU rõ ràng
        params = {k: v for k, v in params.items() if k not in ("device_type", "device", "task_type")}
        params.update({"lgbm": {"device_type": "cpu"}, "xgb": {"device": "cpu"}, "xgbrf": {"device": "cpu"}, "cat": {"task_type": "CPU"},
                       "lstm": {"device": "cpu"}}.get(name, {}))
    return make_model(name, params, allow_cpu=allow_cpu)


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
    say(f"calibrate {args.model} trên {args.colset} ({len(colset.names)} cột) — ES một lần (seed {cfg.seeds[0]})")
    cal = calibrate(store, model, colset, folds, seed=cfg.seeds[0], keep_states=False)
    rounds = rounds_from(cal) if getattr(model, "supports_rounds", True) else None
    say(f"số vòng cố định: {rounds}")
    eps, runs = seed_noise(store, model, colset, folds, rounds, cfg.seeds, cfg.eps_floor_pp, keep_states=False)
    base = runs[0]
    out = {"model": args.model, "tag": args.colset, "colset": colset.to_dict(), "rounds": rounds, "eps": eps,
           "rmse": base.rmse.tolist(), "e0": base.e0.tolist(), "best_iters_es": cal.best_iters.tolist(), "folds": cal.fold_names,
           "seed_rmse": [r.rmse.tolist() for r in runs], "config_hash": cfg.hash()}
    path = cfg.exp_dir / "calib" / f"{args.model}_{args.colset.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    exp_id = new_exp_id("calibrate", args.model, args.colset)
    _log(cfg, exp_id=exp_id, step="calibrate", model=args.model, seed=cfg.seeds[0], colset=args.colset, rounds=json.dumps(rounds),
         **_summ_row(base, base.e0, "E0"), decision=f"eps={eps:.4f}", train_device=getattr(model, "train_device", ""))
    save_run(cfg.exp_dir, exp_id, {**base.to_dict(), "step": "calibrate", "tag": args.colset, "eps": eps, "rounds_fixed": rounds,
                                   "best_iters_es": cal.best_iters.tolist(), "seed_rmse": out["seed_rmse"], "config_hash": cfg.hash()})
    say(f"ε_{args.model} = {eps:.4f} pp → {path}")
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
    base_rmse = np.asarray(cal["rmse"])
    model = model_for(cfg, "lgbm", args.allow_cpu)
    names = list(store.b0_names)
    if args.max_cols:
        names = names[: args.max_cols]
    colset = ColSet(tuple(names))
    say(f"(a) PI: run baseline ES giữ state rồi xáo {len(names)} cột × 3 lần trong VAL")
    base_run = run_config(store, model, colset, folds, rounds=None, seed=cfg.seeds[0], keep_states=True)
    pi = median_over_folds(permutation_importance(store, base_run, list(range(len(names))), repeats=3, seed=cfg.seeds[0]))
    say(f"(b) SA: {len(names)} model 1 cột (ES) — vs E0 và vs B0-306")
    sa_e0, sa_b0 = standalone_gain(store, model, folds, names, cfg.seeds[0], base_rmse,
                                   progress=lambda k, n, nm: say(f"  SA {k}/{n} {nm}") if k % 25 == 0 else None)
    say("(c) MI trên FIT (null xáo trộn)")
    mi = mutual_info(store, folds, colset, seed=cfg.seeds[0])
    table = FilterTable(names, pi, sa_e0, sa_b0, mi)
    df = table.to_frame()
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.exp_dir / "b0_filter.csv", index=False)
    red = df[(df[[f"SA_gain_b0306_h{h}" for h in (1, 2, 3)]] > eps).sum(axis=1) >= 2]
    if len(red):
        say(f"CỜ ĐỎ: {len(red)} cột đơn thắng B0-306 (≥ 2/3 horizon) — B0 bị nhiễu chi phối; R3/R4 sẽ tự thắng ở kiểm chứng")
    sets = table.sets()
    say("kiểm chứng R1–R4 (15fixed_306): " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))
    vdf, chosen, runs = verify_sets(store, model, sets, folds, rounds, base_rmse, eps, seed=cfg.seeds[0])
    vdf.to_csv(cfg.exp_dir / "b0_sets.csv", index=False)
    star = ColSet(tuple(sets[chosen])) if chosen in sets else colset
    star.save(cfg.exp_dir / "b0_star.json")
    for _, r in vdf.iterrows():
        exp_id = new_exp_id("filter_b0", "lgbm", str(r["set"]))
        extra = _summ_row(runs[r["set"]], base_rmse, "B0-306") if r["set"] in runs else {"base": "B0-306"}
        _log(cfg, exp_id=exp_id, step="filter_b0", model="lgbm", seed=cfg.seeds[0], colset=str(r["set"]), rounds="15fixed_306", **extra,
             decision="B0*" if r["chosen"] else "", note=f"n_cols={r['n_cols']}", train_device=getattr(model, "train_device", ""))
        if r["set"] in runs:
            save_run(cfg.exp_dir, exp_id, {**runs[r["set"]].to_dict(), "step": "filter_b0", "set": str(r["set"]), "base": "B0-306", "eps": eps})
    say(f"B0* = {chosen} ({len(star.names)} cột) → experiments/b0_star.json")


def _standalone_factory(store, folds, allow_cpu, cfg):
    lgbm = model_for(cfg, "lgbm", allow_cpu)

    def fn(cand) -> float:
        cs = ColSet((), tuple(cand.columns))
        run = run_config(store, lgbm, cs, folds, rounds=None, seed=cfg.seeds[0], keep_states=False)
        return float(np.median(run.gain_vs(run.e0)))

    return fn


def cmd_loop(cfg: RunConfig, args) -> None:
    """§2.1 cho model m từ B0*: calibrate riêng → add-one 39 candidate → prune PI → 3 seed → win_m → latency → §3 champion + figure."""
    gate(cfg, args, [args.model])
    store, folds, _, _ = load_store(cfg)
    base = colset_from_arg(store, cfg, "b0star")
    model = model_for(cfg, args.model, args.allow_cpu)
    mname = args.model
    exp = cfg.exp_dir
    if load_champion(exp / "champion.json") is None and mname != "lgbm":
        sys.exit("§3: champion ban đầu phải là LightGBM code gốc — chạy `loop --model lgbm` trước.")
    # phase B calibrate riêng trên B0*
    say(f"[{mname}] calibrate trên B0* ({len(base.names)} cột)")
    cal = calibrate(store, model, base, folds, seed=cfg.seeds[0], keep_states=False)
    rounds = rounds_from(cal) if getattr(model, "supports_rounds", True) else None
    eps, runs = seed_noise(store, model, base, folds, rounds, cfg.seeds, cfg.eps_floor_pp, keep_states=False)
    base_run = runs[0]
    say(f"[{mname}] rounds={rounds} ε={eps:.4f} pp; base MedianGain vs E0 = {np.median(base_run.gain_vs(base_run.e0)):+.4f}")
    (exp / "calib").mkdir(parents=True, exist_ok=True)
    (exp / "calib" / f"{mname}_b0star.json").write_text(json.dumps({"model": mname, "rounds": rounds, "eps": eps, "rmse": base_run.rmse.tolist(),
                                                                     "e0": base_run.e0.tolist(), "colset": base.to_dict()}, indent=1), encoding="utf-8")
    # add-one loop
    cands = candidates_from(cfg, args.max_candidates)
    standalone_fn = None if args.no_standalone else _standalone_factory(store, folds, args.allow_cpu, cfg)
    kd_path = exp / f"keepdrop_{mname}.csv"

    def on_row(row, run):
        exp_id = new_exp_id("loop", mname, row["candidate"])
        row["exp_id"] = exp_id  # vào keepdrop_<m>.csv (§7.2)
        say(f"[{mname}] {row['order']:02d} {row['candidate']:<28} Median {row['MedianGain_vs_S']:+.4f} → {row['decision']} (|S|={row['size_S_after']})")
        _log(cfg, exp_id=exp_id, step="loop", model=mname, seed=cfg.seeds[0], colset=row["columns"], n_cols=len(run.colset.names),
             rounds=json.dumps(rounds), base="S_m", MedianGain=row["MedianGain_vs_S"], WinRate=row["WinRate"], P10Gain=row["P10Gain"],
             WorstGain=row["WorstGain"], rmse_cells=row["rmse_cells"], mae_cells=_cells(run.mae), e0_cells=_cells(run.e0),
             gain_cells=row["gain_cells_vs_S"], decision=row["decision"], train_device=getattr(model, "train_device", ""))
        save_run(exp, exp_id, {**run.to_dict(), "step": "loop", "candidate": row["candidate"], "decision": row["decision"], "eps": eps,
                               "MedianGain_vs_S": row["MedianGain_vs_S"]})

    lr = add_one_loop(store, model, base, base_run.rmse, cands, folds, rounds, eps, cfg.seeds[0], base_run.e0, standalone_fn, on_row)
    lr.table.to_csv(kd_path, index=False)
    say(f"[{mname}] F*_m: {len(lr.kept)} KEEP / {len(lr.dropped)} DROP → {kd_path}")
    # prune PI
    pruned, pi_df = prune_pi(store, model, lr.final, folds, rounds, cfg.seeds[0])
    pi_df.to_csv(exp / f"prune_pi_{mname}.csv", index=False)
    say(f"[{mname}] prune PI: giữ {len(pruned.ext)}/{len(lr.final.ext)} cột ext")
    # confirmation 3 seed → win
    unp = confirm(store, model, lr.final, folds, cfg.seeds, keep_states=True)
    prn = confirm(store, model, pruned, folds, cfg.seeds, keep_states=True) if pruned.ext != lr.final.ext else unp
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
                   "best_iters_by_seed": [b.tolist() for b in win.best_iters], "seeds": list(cfg.seeds), "which": which, "folds": [f.name for f in folds],
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
    # §3 champion
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
        plots.fig_path(store, picks, [(f"win = {LABEL.get(mname, mname)}", win_preds, plots.WIN_STYLE[0], plots.WIN_STYLE[1]),
                                      (f"champion = {LABEL.get(champ_label, champ_label)}", champ_preds, plots.CHAMP_STYLE[0], plots.CHAMP_STYLE[1])],
                       exp / "summary" / f"fig_path_{mname}_vs_champion.png",
                       f"Fig P — forecast path win vs champion ({mname} vs {champ_label}): x = t → t+3, y = thay đổi giá so với C_t")
        footer = f"win vs champion: {decision}" + ("" if champ is None else f" — MedianGain {row['MedianGain_vs_champion']:+.4f} (ε {champ['eps']:.4f})")
        plots.fig_hm(gain_pp(win.rmse_mean, win.e0), champ_tab_e0, [f.name.split('_')[-1] for f in folds], LABEL.get(mname, mname),
                     LABEL.get(champ_label, champ_label), footer, exp / "summary" / f"fig_HM_{mname}_vs_champion.png")
    except Exception as e:
        say(f"[{mname}] figure bỏ qua: {e}")


def cmd_ensemble(cfg: RunConfig, args) -> None:
    """§3 ensemble: thành viên = champion + mọi win_m có MedianGain vs E0 > 0; (a) đều, (b) 1/MSE; so với champion."""
    store, folds, _, _ = load_store(cfg)
    exp = cfg.exp_dir
    champ = load_champion(exp / "champion.json")
    if champ is None:
        sys.exit("Chưa có champion.")
    wins = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (exp / "wins").glob("*.json")}
    members = [m for m, w in wins.items() if w["median_gain_vs_e0"] > 0 or m == champ["model"]]
    if len(members) < 2:
        say("Không đủ 2 thành viên — không ensemble.")
        return
    preds = {m: [load_preds(exp / "wins" / f"{m}_seed{k}.npz") for k in range(len(wins[m]["seeds"]))] for m in members}
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
        model = model_for(cfg, mname, args.allow_cpu)
        run = run_config(store, model, cs, [final], rounds=None, seed=cfg.seeds[0], keep_states=True)
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
        _log(cfg, exp_id=exp_id, step="final", model=mname, seed=cfg.seeds[0], colset=key, rounds="ES", **_summ_row(run, run.e0, "E0"),
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
                    seeds=(1, 2, 3), experiments_dir="experiments", require_gpu=False, root=str(out), candidates=[c.name for c in CANDIDATES])
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
    sub.add_parser("ensemble", parents=[common(False)])
    s = sub.add_parser("final", parents=[common(False)]); s.add_argument("--latency-origins", type=int, default=None)
    s = sub.add_parser("smoke-e2e", parents=[common(False)]); s.add_argument("--out", default="tmp_smoke"); s.add_argument("--days", type=float, default=6)
    args = p.parse_args(argv)
    if args.cmd == "smoke-e2e":
        cmd_smoke_e2e(None, args)
        return
    cfg = RunConfig.load(args.config)
    {"check-data": cmd_check_data, "calibrate": cmd_calibrate, "filter-b0": cmd_filter_b0, "loop": cmd_loop, "ensemble": cmd_ensemble,
     "final": cmd_final}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
