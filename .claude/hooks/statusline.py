"""Statusline: `<model> | ctx <used>% | <PHASE> | training <STATE>`.

Nhận JSON trên stdin (schema statusLine cua Claude Code >= 2.1.x, co
context_window.used_percentage); PHASE/TRAINING doc tu 2 dong header cua
.claude/MEMORY.md. Fail-safe: moi loi -> in chuoi toi thieu, exit 0.
"""
import json
import pathlib
import sys


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    parts = []

    model = (data.get("model") or {}).get("display_name")
    if model:
        parts.append(str(model))

    ctx = data.get("context_window") or {}
    pct = ctx.get("used_percentage")
    if isinstance(pct, (int, float)):
        parts.append(f"ctx {pct:.0f}%")

    phase, training = None, None
    try:
        cwd = pathlib.Path(data.get("cwd") or ".")
        mem = cwd / ".claude" / "MEMORY.md"
        if mem.exists():
            for line in mem.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]:
                if line.startswith("PHASE:"):
                    phase = line.split(":", 1)[1].strip()
                elif line.startswith("TRAINING:"):
                    training = line.split(":", 1)[1].strip()
    except Exception:
        pass

    if phase:
        parts.append(phase)
    if training:
        parts.append(f"training {training}")

    print(" | ".join(parts) if parts else "P0_forecasting")


if __name__ == "__main__":
    main()
