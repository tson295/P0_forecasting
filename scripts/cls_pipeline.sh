#!/bin/bash
# End-to-end classification sweep, meant to run inside tmux:
#   tmux new -s classification 'scripts/cls_pipeline.sh baseline'
#
# baseline: stage 1 (18 multi-horizon jobs) -> stage-1 report + upload -> stage 2
#           (54 single-horizon jobs) -> leakage audit -> report -> upload -> read-back
# followup: jobs registered with stage "followup" -> audit -> report -> upload -> verify
# Every step is idempotent: completed jobs are skipped, failed ones are retried by the
# scheduler (resume from their last epoch), and a re-run picks up where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/.hf_home"
source ~/venv_cls/bin/activate
CSV="${CSV:-$HOME/data/BTC_L10_gate_1y.csv}"
LOG=runs/cls/logs/pipeline.log
mkdir -p runs/cls/logs
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() { say "BEGIN $*"; "$@" 2>&1 | grep --line-buffered -v NNPACK | tee -a "$LOG"; local rc=${PIPESTATUS[0]}; say "END rc=$rc $*"; return $rc; }

MODE="${1:-baseline}"
if [ "$MODE" = "baseline" ]; then
  step python scripts/cls_registry.py init-baseline || exit 1
  step python scripts/cls_scheduler.py --stages multi --csv "$CSV" || { say "stage multi has failed jobs; stopping"; exit 1; }
  step python scripts/cls_report.py --stage multi || say "stage-1 report failed (the final report is regenerated anyway)"
  # A stuck or failing interim upload must not hold stage 2: the final upload resends everything.
  step timeout -k 60 3h python scripts/cls_upload.py upload --only-runs || say "stage-1 upload failed/timed out; continuing"
  step python scripts/cls_scheduler.py --stages single --csv "$CSV" || { say "stage single has failed jobs; stopping"; exit 1; }
elif [ "$MODE" = "followup" ]; then
  step python scripts/cls_scheduler.py --stages followup --csv "$CSV" || { say "follow-up has failed jobs; stopping"; exit 1; }
fi
step python scripts/cls_audit.py --csv "$CSV" || { say "LEAKAGE AUDIT FAILED: results are not valid"; exit 1; }
step python scripts/cls_report.py || { say "REPORT FAILED"; exit 1; }
step timeout -k 60 8h python scripts/cls_upload.py upload || { say "UPLOAD FAILED; re-run scripts/cls_pipeline.sh $MODE"; exit 1; }
step python scripts/cls_upload.py verify || { say "HF READ-BACK FAILED"; exit 1; }
say "PIPELINE_DONE $MODE"
