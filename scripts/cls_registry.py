"""Create or inspect the run registry (runs/cls/registry.json).

  python scripts/cls_registry.py init-baseline     # register the 72 required jobs
  python scripts/cls_registry.py status            # counts per stage and status
  python scripts/cls_registry.py reset-failed      # failed -> planned (manual retry)
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cls.jobs import Registry, baseline_jobs

DEFAULT = ROOT/"runs/cls/registry.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init-baseline", "status", "reset-failed"))
    p.add_argument("--registry", default=str(DEFAULT))
    args = p.parse_args()
    registry = Registry(args.registry)
    if args.command == "init-baseline":
        jobs = baseline_jobs()
        assert len(jobs) == 72 and len({j["id"] for j in jobs}) == 72
        added = registry.add(jobs)
        print(f"registered {len(added)} new baseline jobs ({72-len(added)} already present)")
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
