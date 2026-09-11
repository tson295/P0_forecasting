#!/usr/bin/env bash
# Exact training command of the Zenodo OB run on the Vast instance (GPU-only; no smoke/probe/warmup).
# summarize runs only after train exits 0. HF_HOME is overridden because /etc/environment points it at
# /workspace/.hf_home (root-owned, not writable by this user); the pinned TimesFM checkpoint is cached there.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export P0_OB_VAST=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/ubuntu/.cache/huggingface
LOG=experiments/orderbook_zenodo/logs
CFG=configs/orderbook_zenodo.json
echo "train start $(date -u +%FT%TZ)" | tee -a "$LOG/train.log"
/home/ubuntu/venv-ob/bin/python -m src_OB train --config "$CFG" 2>&1 | tee -a "$LOG/train.log"
status=${PIPESTATUS[0]}
echo "TRAIN_EXIT=$status $(date -u +%FT%TZ)" | tee -a "$LOG/train.log"
if [ "$status" -eq 0 ]; then
    /home/ubuntu/venv-ob/bin/python -m src_OB summarize --config "$CFG" 2>&1 | tee -a "$LOG/summarize.log"
    echo "SUMMARIZE_EXIT=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$LOG/summarize.log"
fi
