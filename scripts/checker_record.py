"""Ghi finding của agent `checker` vào experiments/<run>/checker_log.jsonl — KHÔNG tương tác (quyết định 2026-09-04).

    python scripts/checker_record.py --exp experiments/full --stage pre-run --model lgbm --severity WARN \
        --check-id CORR_HIGH --message "..." [--file src/p0/x.py] [--ref "L123"]
    python scripts/checker_record.py --exp experiments/full --blocking          # liệt kê ERROR chưa đóng (exit 1 nếu có)

Severity: PASS / INFO / WARN / ERROR. ERROR = bất biến cứng → run bị chặn cho tới khi sửa và ghi PASS cùng check_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from p0.checker_log import SEVERITIES, blocking_errors, record  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(ROOT / "experiments" / "full"))
    ap.add_argument("--stage", default="code-review")
    ap.add_argument("--model", default="")
    ap.add_argument("--severity", choices=SEVERITIES, default="INFO")
    ap.add_argument("--check-id", default="")
    ap.add_argument("--message", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--blocking", action="store_true", help="chỉ liệt kê ERROR chưa đóng; exit 1 nếu còn")
    a = ap.parse_args()
    if a.blocking:
        errs = blocking_errors(a.exp)
        for e in errs:
            print(json.dumps(e, ensure_ascii=False))
        print(f"{len(errs)} ERROR chưa đóng")
        sys.exit(1 if errs else 0)
    if not a.check_id or not a.message:
        sys.exit("cần --check-id và --message")
    row = record(a.exp, a.stage, a.severity, a.check_id, a.message, a.model, a.file, a.ref)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
