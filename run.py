"""Entry point: python run.py <subcommand> ... (xem src/p0/cli.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):  # console Windows cp1252 không in được "→"; Linux/Vast không ảnh hưởng
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

from p0.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
