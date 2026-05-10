#!/bin/bash
# Phase 5 step 6→7 helper: evaluate Tongji TPIR@5% gate (V17.10) after
# Tongji 50ep CCNet completes; if pass, launch IITD CCNet 50ep.
#
# Usage:
#   bash experiments/run_phase5_iitd_after_tongji_gate.sh
#
# Pre-requisite: Tongji CCNet 50ep training has completed and saved
#   experiments/generated/tongji_full/ccnet_arcface_tongji_scratch_128.pt
#
# Behavior:
#   1. Extract CCNet embeddings on Tongji manifest
#   2. Run single-seed orchestrator (order_seed=0, enroll_seed=0,
#      target_fpir=0.05) at endpoint t_step=t_max
#   3. Read protocol_c_recal.json and parse final-step TPIR
#   4. Apply V17.10 gate:
#        TPIR ≥ 0.50  → tongji_gate_result=passed   → launch IITD
#        0.30..0.50   → tongji_gate_result=marginal → manual decision
#        TPIR < 0.30  → tongji_gate_result=failed   → manual decision
#   5. (passed only) launch IITD 50ep training in background

set -e
cd "/Users/kimeunsu/Desktop/Research Notebook/secure/Secure"

CKPT="experiments/generated/tongji_full/ccnet_arcface_tongji_scratch_128.pt"
NPZ_OUT="experiments/generated/tongji_full/ccnet_tongji_128_embeddings.npz"
PROTOCOL_DIR="experiments/generated/tongji_full/ccnet_protocols"

if [ ! -f "$CKPT" ]; then
  echo "[FAIL] Tongji CCNet checkpoint not found at $CKPT" >&2
  echo "       Wait for the 50ep training to complete before running this helper." >&2
  exit 1
fi

echo "[step 6.1] Extracting Tongji CCNet embeddings..."
PYTHONPATH=. python -m exp1_baselines.extract_embeddings \
  --model ccnet \
  --checkpoint "$CKPT" \
  --manifest_dir experiments/generated/tongji_full \
  --output_npz "$NPZ_OUT"

echo "[step 6.2] Running single-seed orchestrator at endpoint..."
mkdir -p "$PROTOCOL_DIR"
PYTHONPATH=. python -m exp1_baselines.eval.orchestrator \
  --embeddings "$NPZ_OUT" \
  --manifest experiments/generated/tongji_full/manifest.csv \
  --out "$PROTOCOL_DIR" \
  --target_fpir 0.05 \
  --t_step 10 \
  --order_seed 0 \
  --enroll_seed 0 \
  --K 3

echo "[step 6.3] Evaluating V17.10 TPIR@5% gate..."
TPIR_FINAL=$(python3 -c "
import json
d = json.load(open('$PROTOCOL_DIR/protocol_c_recal.json'))
if isinstance(d, dict) and 'steps' in d:
    d = d['steps']
final = d[-1]
print(final['tpir'])
")
echo "[Tongji CCNet endpoint TPIR@5% = $TPIR_FINAL]"

# V17.10 gate
GATE_RESULT=$(python3 -c "
tpir = $TPIR_FINAL
if tpir >= 0.50: print('passed')
elif tpir >= 0.30: print('marginal')
else: print('failed_undertraining')
")
echo "[V17.10 gate result = $GATE_RESULT]"

# Save gate-time metadata per V17.11
cat > "$PROTOCOL_DIR/gate_metadata.json" <<EOF
{
  "tpir_gate_alpha": 0.05,
  "tpir_gate_value": $TPIR_FINAL,
  "tpir_gate_result": "$GATE_RESULT",
  "stage": "L1_50ep",
  "dataset": "tongji"
}
EOF
echo "[saved] $PROTOCOL_DIR/gate_metadata.json"

if [ "$GATE_RESULT" = "passed" ]; then
  echo ""
  echo "[step 7] Tongji passed; launching IITD CCNet 50ep training in background..."
  rm -f "experiments/generated/iitd_full/ccnet_arcface_iitd_scratch_128.pt" \
        "experiments/generated/iitd_full/ccnet_train.log"
  nohup env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u -m exp1_baselines.train_arcface \
    --config exp1_baselines/configs/ccnet_arcface_iitd_128.yaml \
    > experiments/generated/iitd_full/ccnet_train.log 2>&1 &
  IITD_PID=$!
  echo "[IITD training launched in background, PID=$IITD_PID]"
  echo "[IITD log: experiments/generated/iitd_full/ccnet_train.log]"
elif [ "$GATE_RESULT" = "marginal" ]; then
  echo ""
  echo "[V17.10 marginal] TPIR in [0.30, 0.50). Manual decision required:"
  echo "  Option A: extend Tongji to 200ep (L2 fallback) before IITD"
  echo "  Option B: launch IITD 50ep anyway and consider both 50ep results as marginal"
  echo "Re-run this script with --force-iitd to launch IITD anyway."
else
  echo ""
  echo "[V17.10 failed_undertraining] TPIR < 0.30."
  echo "  L2 fallback recommended: train Tongji for 200ep before IITD."
  echo "  See plan §V17.10 for the exclusion path."
fi
