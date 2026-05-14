#!/bin/bash
# V19 Path 3b — Tongji CCM-on-MFN training chained after IR50 completes.
# This helper polls for the IR50 checkpoint and, once present, launches the
# CCM-on-MFN training (sequential, to avoid MPS memory contention per V17
# PHASE5_SESSION_LOG.md §6.1 lesson 1).
#
# Usage (run AFTER ir50 training is already launched in the background):
#   bash experiments/run_phase5_v19_tongji_after_ir50.sh
#
# Pre-requisite: ir50 training started, e.g.
#   nohup env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u \
#     -m exp1_baselines.train_arcface \
#     --config exp1_baselines/configs/ir50_arcface_tongji_112.yaml \
#     > experiments/generated/tongji_full/ir50_train.log 2>&1 &

set -e
cd "/Users/kimeunsu/Desktop/Research Notebook/secure/Secure"

IR50_CKPT="experiments/generated/tongji_full/ir50_arcface_tongji_scratch_112.pt"
IR50_LOG="experiments/generated/tongji_full/ir50_train.log"
CCM_LOG="experiments/generated/tongji_full/ccm_mfn_train.log"

# Poll for IR50 completion. The training script saves ckpt only on
# `epochs_completed == epochs_planned`, so ckpt timestamp updates only at the
# very end. We watch for the marker "epochs_completed: 50" in the log, OR for
# the [save] line.

echo "[chain] Waiting for IR50 Tongji 50ep training to complete..."
echo "[chain] Polling for completion marker in $IR50_LOG every 60s"

while true; do
  if [ -f "$IR50_LOG" ] && grep -q "\[save\] checkpoint" "$IR50_LOG"; then
    break
  fi
  # also bail out if training process died without saving
  if ! pgrep -f "exp1_baselines.train_arcface.*ir50_arcface_tongji" > /dev/null; then
    if [ -f "$IR50_LOG" ] && grep -q "\[save\] checkpoint" "$IR50_LOG"; then
      break
    fi
    echo "[chain] IR50 training process is no longer running and no [save] line found."
    echo "[chain] Inspect $IR50_LOG for failure cause."
    tail -20 "$IR50_LOG" 2>&1
    exit 1
  fi
  sleep 60
done

echo ""
echo "[chain] IR50 Tongji 50ep training completed."
tail -5 "$IR50_LOG"

# Sanity: ckpt file present
if [ ! -f "$IR50_CKPT" ]; then
  echo "[ERR] $IR50_CKPT not present despite [save] marker. Abort."
  exit 1
fi

echo ""
echo "[chain] Launching CCM-on-MFN Tongji 50ep training in background..."
mkdir -p experiments/generated/tongji_full/ccm_mfn_logs
rm -f "$CCM_LOG"
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u -m exp1_baselines.train_arcface \
  --config exp1_baselines/configs/ccm_mfn_arcface_tongji_112.yaml \
  > "$CCM_LOG" 2>&1 &
CCM_PID=$!
echo "[chain] CCM-on-MFN launched, PID=$CCM_PID"
echo "[chain] log: $CCM_LOG"
echo "[chain] After CCM-on-MFN finishes, run:"
echo "  bash experiments/run_phase5_v19_gate.sh tongji all"
