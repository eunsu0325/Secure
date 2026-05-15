#!/bin/bash
# V19 Path 3b — V17.10 TPIR@5% gate runner for the three V19 backbones on a
# single dataset. Per V17.7 + V19.6, applies gate per (backbone, dataset).
#
# Usage:
#   bash experiments/run_phase5_v19_gate.sh <dataset>          # tongji | iitd
#   bash experiments/run_phase5_v19_gate.sh tongji mfn         # single backbone
#   bash experiments/run_phase5_v19_gate.sh tongji all         # all three (default)
#
# Behavior:
#   For each backbone in {mfn, ir50, ccm_mfn} (or the single one specified):
#     1. Verify the ckpt exists at experiments/generated/<dataset>_full/<arch>_arcface_<dataset>_scratch_<size>.pt
#     2. Extract embeddings -> <arch>_<dataset>_<size>_embeddings.npz
#     3. Run single-seed orchestrator (order_seed=0, enroll_seed=0,
#        target_fpir=0.05) at endpoint t_step
#     4. Parse protocol_c_recal.json final-step TPIR
#     5. Save gate_metadata.json with verdict (passed | marginal | failed)

set -e
cd "/Users/kimeunsu/Desktop/Research Notebook/secure/Secure"

DATASET="${1:-tongji}"
WHICH="${2:-all}"

case "$DATASET" in
  tongji)         FULL_DIR="experiments/generated/tongji_full"; SIZE=112 ;;
  iitd)           FULL_DIR="experiments/generated/iitd_full"; SIZE=112 ;;
  xjtu_up_huawei) FULL_DIR="experiments/generated/xjtu_up_huawei_full"; SIZE=112 ;;
  *)
    echo "[FAIL] unknown dataset '$DATASET' (expected tongji|iitd|xjtu_up_huawei)" >&2
    exit 1 ;;
esac

if [ "$WHICH" = "all" ]; then
  ARCHS=("mfn" "ir50" "ccm_mfn")
else
  ARCHS=("$WHICH")
fi

run_one() {
  local arch="$1"
  local dataset="$2"
  local full_dir="$3"
  local size="$4"

  local ckpt="${full_dir}/${arch}_arcface_${dataset}_scratch_${size}.pt"
  local npz="${full_dir}/${arch}_${dataset}_${size}_embeddings.npz"
  local proto_dir="${full_dir}/${arch}_protocols"

  echo ""
  echo "============================================================"
  echo "[V19 gate: ${arch} on ${dataset}]"
  echo "============================================================"

  if [ ! -f "$ckpt" ]; then
    echo "[FAIL] checkpoint not found at $ckpt — wait for training to complete." >&2
    return 1
  fi
  echo "  ckpt: $ckpt"
  echo "  npz : $npz"

  echo "[step 1] Extracting embeddings (--model ${arch})..."
  PYTHONUNBUFFERED=1 PYTHONPATH=. python -u -m exp1_baselines.extract_embeddings \
    --model "$arch" \
    --checkpoint "$ckpt" \
    --manifest_dir "$full_dir" \
    --output_npz "$npz"

  mkdir -p "$proto_dir"
  echo "[step 2] Running single-seed orchestrator (target_fpir=0.05, t_step=10)..."
  PYTHONUNBUFFERED=1 PYTHONPATH=. python -u -m exp1_baselines.eval.orchestrator \
    --embeddings "$npz" \
    --manifest "${full_dir}/manifest.csv" \
    --out "$proto_dir" \
    --target_fpir 0.05 \
    --t_step 10 \
    --order_seed 0 \
    --enroll_seed 0 \
    --K 3

  echo "[step 3] Parsing TPIR@5% from protocol_c_recal.json..."
  local tpir_final
  tpir_final=$(python3 -c "
import json
d = json.load(open('${proto_dir}/protocol_c_recal.json'))
if isinstance(d, dict) and 'steps' in d:
    d = d['steps']
final = d[-1]
print(final['tpir'])
")
  echo "  [${arch}/${dataset}] endpoint TPIR@5% = ${tpir_final}"

  local verdict
  verdict=$(python3 -c "
tpir = ${tpir_final}
if tpir >= 0.50: print('passed')
elif tpir >= 0.30: print('marginal')
else: print('failed_undertraining')
")
  echo "  [V17.10 gate: ${verdict}]"

  cat > "${proto_dir}/gate_metadata.json" <<EOF
{
  "tpir_gate_alpha": 0.05,
  "tpir_gate_value": ${tpir_final},
  "tpir_gate_result": "${verdict}",
  "stage": "L1_50ep",
  "dataset": "${dataset}",
  "backbone": "${arch}",
  "v19_path": "3b"
}
EOF
  echo "  [saved] ${proto_dir}/gate_metadata.json"
}

for arch in "${ARCHS[@]}"; do
  run_one "$arch" "$DATASET" "$FULL_DIR" "$SIZE" || {
    echo "[ERR] gate for ${arch} on ${DATASET} failed; continuing with remaining backbones"
    continue
  }
done

echo ""
echo "============================================================"
echo "[V19 gate summary for ${DATASET}]"
echo "============================================================"
for arch in "${ARCHS[@]}"; do
  local_meta="${FULL_DIR}/${arch}_protocols/gate_metadata.json"
  if [ -f "$local_meta" ]; then
    tpir=$(python3 -c "import json; d=json.load(open('${local_meta}')); print(d['tpir_gate_value'])")
    verdict=$(python3 -c "import json; d=json.load(open('${local_meta}')); print(d['tpir_gate_result'])")
    printf "  %-10s TPIR@5%%=%.4f  ->  %s\n" "$arch" "$tpir" "$verdict"
  else
    printf "  %-10s [NO GATE METADATA]\n" "$arch"
  fi
done
