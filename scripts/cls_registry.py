"""Create or inspect the run registry (runs/cls/registry.json).

  python scripts/cls_registry.py init-baseline     # register the 72 required jobs
  python scripts/cls_registry.py status            # counts per stage and status
  python scripts/cls_registry.py reset-failed      # failed -> planned (manual retry)
  python scripts/cls_registry.py add-followup --experiment bins --variant k64 \
      --method equal_width --label-set labeling/equal_width_k64_p99.json \
      --archs transformer --formulation multi --folds 1,2,3 [--model-kwargs '{...}']

A follow-up job changes exactly one factor against its baseline twin (same method,
architecture, formulation, horizons and fold): the label set (bins, range, overflow)
or a model_kwargs capacity override. Everything else is the baseline recipe.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cls import HORIZONS
from src.cls.jobs import Registry, baseline_jobs, make_job

DEFAULT = ROOT/"runs/cls/registry.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init-baseline", "status", "reset-failed", "add-followup"))
    p.add_argument("--registry", default=str(DEFAULT))
    p.add_argument("--experiment")
    p.add_argument("--variant")
    p.add_argument("--method", choices=("equal_width", "quantile"))
    p.add_argument("--label-set", help="Frozen label JSON relative to the repo root")
    p.add_argument("--archs", default="ofi_lstm,moderntcn,transformer")
    p.add_argument("--formulation", choices=("multi", "single"), default="multi")
    p.add_argument("--horizons", default="60,120,180", help="single formulation: one job per horizon")
    p.add_argument("--folds", default="1,2,3")
    p.add_argument("--model-kwargs", default="{}")
    p.add_argument("--hypothesis", default="")
    args = p.parse_args()
    registry = Registry(args.registry)
    if args.command == "init-baseline":
        jobs = baseline_jobs()
        assert len(jobs) == 72 and len({j["id"] for j in jobs}) == 72
        added = registry.add(jobs)
        print(f"registered {len(added)} new baseline jobs ({72-len(added)} already present)")
    elif args.command == "add-followup":
        if not (args.experiment and args.variant and args.method):
            raise SystemExit("add-followup needs --experiment, --variant and --method")
        jobs = []
        horizon_sets = ([list(HORIZONS)] if args.formulation == "multi"
                        else [[int(h)] for h in args.horizons.split(",")])
        for arch in args.archs.split(","):
            for hs in horizon_sets:
                for fold in (int(f) for f in args.folds.split(",")):
                    jobs.append(make_job(args.method, args.formulation, arch, hs, fold, stage="followup",
                                         experiment=args.experiment, variant=args.variant,
                                         label_set_path=args.label_set, model_kwargs=json.loads(args.model_kwargs),
                                         extra=dict(hypothesis=args.hypothesis)))
        added = registry.add(jobs)
        print(f"registered {len(added)} follow-up jobs: {[j['id'] for j in added]}")
    elif args.command == "reset-failed":
        with registry.locked() as data:
            for job in data["jobs"]:
                if job["status"] == "failed":
                    job["status"], job["retried"] = "planned", True
                    print("reset", job["id"])
    data = registry.read()
    counts = Counter((j["stage"], j["status"]) for j in data["jobs"])
    print(json.dumps({f"{s}:{st}": n for (s, st), n in sorted(counts.items())}, indent=1))


if __name__ == "__main__":
    main()
