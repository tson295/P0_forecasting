#!/bin/bash
# Register the follow-up experiments (run ONLY after the 72-job baseline is complete).
# Each follow-up changes exactly one factor against its baseline twin: equal-width,
# multi-horizon, same architecture, fold, recipe and samples. Label definitions were
# frozen from fold-1 train before any baseline result existed (commit 52eb40d).
#
#   scripts/cls_followups.sh && tmux new -s fu 'scripts/cls_pipeline.sh followup'
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/venv_cls/bin/activate
R() { python scripts/cls_registry.py add-followup "$@"; }

# FU-1 number of bins (bin width). Argmax can never decode "no move": its smallest call
# is +-w/2, and exact-zero moves inflate [0, w). A constant +-w/2 call costs
# (w/2)^2 / MSE_E0 of R2 gain, so the argmax loss should shrink as K grows.
H1="Argmax loss vs E0 is dominated by the +-w/2 quantization of the central bins; more bins (smaller w) move argmax R2 gain toward 0, fewer bins away from it."
R --experiment bins --variant k64  --method equal_width --label-set labeling/equal_width_k64_p99.json  --hypothesis "$H1"
R --experiment bins --variant k128 --method equal_width --label-set labeling/equal_width_k128_p99.json --hypothesis "$H1"
R --experiment bins --variant k16  --method equal_width --label-set labeling/equal_width_k16_p99.json  --archs ofi_lstm,transformer --hypothesis "$H1"

# FU-2 finite range (p98 / p99.5 instead of p99, K = 32). Changes the bin width with the
# range and moves mass into / out of the overflow classes, whose argmax calls decode to
# +-(q + w/2) and dominate squared error when wrong.
H2="A narrower range (p98) shrinks w and the overflow representatives but doubles overflow frequency; a wider range (p99.5) does the opposite. Which effect dominates argmax R2 gain?"
R --experiment range --variant p98  --method equal_width --label-set labeling/equal_width_k32_p98.json  --archs ofi_lstm,transformer --hypothesis "$H2"
R --experiment range --variant p995 --method equal_width --label-set labeling/equal_width_k32_p995.json --archs ofi_lstm,transformer --hypothesis "$H2"

# FU-3 capacity inside ModernTCN. The WF3 50.6M-parameter ModernTCN overfits the class
# target within 1-3 epochs (train CE keeps falling, validation CE rises from epoch 2).
H3="ModernTCN overfits because of its width; a 4x narrower stage width (dims 64) lowers validation CE and improves the decoded metrics."
R --experiment capacity --variant tcn_dims64 --method equal_width --label-set labeling/equal_width_k32_p99.json \
  --archs moderntcn --model-kwargs '{"dims": [64, 64, 64, 64]}' --hypothesis "$H3"

python scripts/cls_registry.py status
