"""Run ONE classification job end to end: train (resuming from last/ if present) ->
select best on validation -> export validation/test predictions and all metrics.

  python scripts/cls_run.py --config runs/cls/<...>/run_config.json

The scheduler (scripts/cls_scheduler.py) writes the config and launches this; it can
also be run by hand to re-run or resume a single failed job.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cls.trainer import run, set_status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, help="Override (smoke tests only)")
    p.add_argument("--run-dir", help="Override the run directory (smoke tests only)")
    args = p.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    run_dir = Path(args.run_dir or cfg["run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT/run_dir
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    try:
        summary = run(cfg, run_dir, ROOT)
    except BaseException as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        set_status(run_dir, state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    print(json.dumps({k: summary[k] for k in ("job_id", "parameters", "best_epoch", "train_seconds")}
                     | {"test": summary["metrics"]["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
