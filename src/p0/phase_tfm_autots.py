"""One real training command for TimesFM and AutoTS only; no probes or tests."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def run_phase(cfg, args):
    from . import fold_parallel
    from .cli import cmd_autots_search, cmd_lock_s0, cmd_loop, cmd_tfm_final, gate, _write_json

    wanted = ["tfm", "autots_wr", "autots_mr"]
    if cfg.phase != "tfm_autots" or cfg.model_order != wanted or not cfg.defer_champion:
        raise ValueError("Use configs/tfm_autots.json; this phase has exactly TimesFM and two AutoTS templates")
    gate(cfg, args, wanted)
    code = hashlib.sha256()
    # All Python implementation files participate: helper edits also invalidate
    # completed-stage reuse. This is provenance, not an executable verification.
    project = Path(cfg.root)
    for source in sorted((project / "src" / "p0").glob("*.py")) + [project / "src_OB" / "gpu.py"]:
        code.update(str(source.relative_to(project)).encode())
        code.update(source.read_bytes())
    code_hash = code.hexdigest()
    # A separate output contract prevents reuse of old residual-first artifacts.
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    contract = cfg.exp_dir / "phase_config.json"
    if contract.exists():
        recorded = json.loads(contract.read_text(encoding="utf-8"))
        if recorded["config_hash"] != cfg.hash() or recorded.get("code_hash") != code_hash:
            raise ValueError("Phase output belongs to different code/config; choose a new experiments_dir")
    else:
        if any(cfg.exp_dir.iterdir()):
            raise ValueError("Nonempty phase output without a provenance contract; choose a new experiments_dir")
        contract.write_text(json.dumps({"config_hash": cfg.hash(), "code_hash": code_hash, "config": cfg.to_dict(),
                                        "method": "tfm-first-heldout-residual-v1/autots-batch-v1/autots-cpu-pool-v1"}, indent=2), encoding="utf-8")
    ns = argparse.Namespace(smoke=False, allow_cpu=False, resume=bool(args.resume),
                            no_standalone=True, max_candidates=None, latency_origins=None,
                            data_config=None, max_rows=None, config=args.config)
    progress_path = cfg.exp_dir / "phase_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed": []}
    if progress_path.exists() and not args.resume:
        raise ValueError("Phase already started; use --resume with the same config")
    progress.pop("error", None)
    progress["status"] = "running"

    def stage(name, action):
        if name in progress["completed"]:
            return
        progress["active"] = name
        _write_json(progress_path, progress)
        action()
        progress["completed"].append(name)
        progress["active"] = None
        _write_json(progress_path, progress)
    # Load existing feature definitions/artifacts, not train the other families.
    if not all((cfg.exp_dir / "s0" / f"{name}.json").exists() for name in wanted):
        cmd_lock_s0(cfg, ns)
    try:
        for name in wanted:
            ns.model = name
            # --resume is only meaningful once this branch has its own calibration.
            ns.resume = bool(args.resume and (cfg.exp_dir / "calib" / f"{name}_base.json").exists())
            stage("loop:" + name, lambda: cmd_loop(cfg, ns))
            if name == "tfm":
                stage("tfm-final", lambda: cmd_tfm_final(cfg, ns))
        stage("autots-search", lambda: cmd_autots_search(cfg, ns))
        summarize_phase(cfg.exp_dir)
        progress["status"] = "completed"
        _write_json(progress_path, progress)
        (cfg.exp_dir / "PHASE_REPORT.md").write_text(
            "# TimesFM / AutoTS phase\n\n"
            "All configured feature-search and final-representative stages completed.\n"
            "See tfm_autots_per_fold_horizon.csv and tfm_autots_summary.csv for results.\n"
            "LoRA uses a FIT prefix and inner early stopping; residual heads use a later held-out FIT suffix.\n"
            "AutoTS WR/MR rolling predictions are batched across independent origins.\n"
            "AutoTS feature-search folds use CPU caches prepared before calibration and candidate fits.\n"
            "MR later recursive steps and native final template bake-off retain their required preprocessing.\n"
            "Native forecast cache timing and real batch timing are stored separately from fit timing.\n"
            "No smoke/tests/probe fits/warmup/benchmark pass or other model families were run by this command.\n"
            "The final TEST holdout was not evaluated in this phase.\n"
            f"\nCode hash: {code_hash}\nConfig hash: {cfg.hash()}\n", encoding="utf-8")
    except BaseException as exc:
        progress["status"] = "failed"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(progress_path, progress)
        raise
    finally:
        fold_parallel.shutdown()


def summarize_phase(output: Path):
    rows = []
    for name in ("tfm", "autots"):
        payload = json.loads((output / "wins" / f"{name}.json").read_text(encoding="utf-8"))
        rmse, e0 = np.asarray(payload["rmse_mean"], float), np.asarray(payload["e0"], float)
        seed_rmse = np.asarray(payload["seed_rmse"], float)
        if rmse.shape != e0.shape or rmse.ndim != 2 or rmse.shape[1] != 3 or not len(rmse):
            raise ValueError(f"Invalid final {name} fold/horizon results")
        if seed_rmse.ndim != 3 or seed_rmse.shape[1:] != rmse.shape:
            raise ValueError(f"Missing per-seed RMSE for {name}")
        if not np.isfinite(seed_rmse).all() or not np.isfinite(rmse).all() or not np.isfinite(e0).all():
            raise ValueError(f"Nonfinite final metrics for {name}")
        for fold in range(len(rmse)):
            for h in range(3):
                denominator = float(e0[fold, h])
                ratio = float(rmse[fold, h]) / denominator if denominator else None
                rows.append({"model": name, "fold": fold + 1, "horizon_minutes": h + 1,
                             "rmse_mean_over_seeds": float(rmse[fold, h]), "rmse_e0": denominator,
                             "rmse_gain_vs_e0": 1 - ratio if ratio is not None else None,
                             "r2_os_vs_e0": (1 - float(np.mean(seed_rmse[:, fold, h] ** 2)) / denominator ** 2)
                             if denominator else None,
                             "aggregation": "mean over seed-specific metrics, then unweighted mean over folds",
                             "e0_status": "ok" if denominator else "zero_denominator"})
    table = pd.DataFrame(rows)
    table.to_csv(output / "tfm_autots_per_fold_horizon.csv", index=False)
    table.groupby(["model", "horizon_minutes"], as_index=False)[
        ["rmse_mean_over_seeds", "rmse_e0", "rmse_gain_vs_e0", "r2_os_vs_e0"]].mean().to_csv(
            output / "tfm_autots_summary.csv", index=False)


def main():
    from .config import RunConfig

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tfm_autots.json")
    p.add_argument("--resume", action="store_true", help="Resume saved feature-search progress with the same contract")
    args = p.parse_args()
    run_phase(RunConfig.load(args.config), args)


if __name__ == "__main__":
    main()
