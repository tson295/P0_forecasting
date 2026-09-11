#!/usr/bin/env bash
# Real phase execution only. No installation, test, probe fit or warmup.
set -euo pipefail
cd "$(dirname "$0")/.."

P0_RUN_CONFIG=${1:-configs/tfm_autots.json}
P0_RUN_PYTHON=${P0_RUN_PYTHON:-.venv/bin/python}
if [ ! -x "$P0_RUN_PYTHON" ]; then
  echo "Missing $P0_RUN_PYTHON; prepare the CUDA environment in docs/VAST_SESSION_PROMPT.md first." >&2
  exit 1
fi
# One real run per checkout, including after SSH reconnects. Do not kill an existing run.
exec 9>"$(git rev-parse --git-path tfm_autots.run.lock)"
if ! flock -n 9; then
  echo "Another phase process owns this checkout; inspect its existing log/tmux session." >&2
  exit 1
fi
export P0_TFM_AUTOTS_VAST=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export P0_GPU_DEVICES=0
export P0_FOLD_WORKERS=1
export P0_GPU_SLOTS_PER_DEVICE=1
export PYTHONUNBUFFERED=1

# Session logs live outside the phase output: that directory must initially be empty.
mkdir -p experiments/tfm_autots_sessions
P0_SESSION_DIR=$(mktemp -d "experiments/tfm_autots_sessions/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
cp "$P0_RUN_CONFIG" "$P0_SESSION_DIR/config.json"
git rev-parse HEAD > "$P0_SESSION_DIR/code_commit.txt"
git diff HEAD -- src/p0 src_OB/gpu.py configs requirements-tfm-autots.txt scripts/vast_tfm_autots_run.sh > "$P0_SESSION_DIR/code_changes.patch"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$P0_SESSION_DIR/gpu.csv"
"$P0_RUN_PYTHON" - "$P0_SESSION_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

# Package metadata only; no library probe or model execution.
Path(sys.argv[1]).write_text(json.dumps({
    "python": sys.version, "platform": platform.platform(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
}, indent=2), encoding="utf-8")
PY
P0_OUTPUT_DIR=$("$P0_RUN_PYTHON" - "$P0_RUN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
cfg = json.loads(path.read_text())
print(Path(cfg.get("root", path.parent.parent)) / cfg["experiments_dir"])
PY
)
P0_RESUME_ARGS=()
if [ -f "$P0_OUTPUT_DIR/phase_progress.json" ]; then
  P0_RESUME_ARGS=(--resume)
fi
echo "Real run log: $P0_SESSION_DIR/training.log"
set +e
"$P0_RUN_PYTHON" -u run.py tfm-autots --config "$P0_RUN_CONFIG" "${P0_RESUME_ARGS[@]}" 2>&1 | tee "$P0_SESSION_DIR/training.log"
P0_PIPE_EXIT=("${PIPESTATUS[@]}")
P0_RUN_EXIT=${P0_PIPE_EXIT[0]}
if [ "$P0_RUN_EXIT" -eq 0 ]; then
  P0_RUN_EXIT=${P0_PIPE_EXIT[1]}
fi
set -e
printf '%s\n' "$P0_RUN_EXIT" > "$P0_SESSION_DIR/exit_code.txt"
exit "$P0_RUN_EXIT"
