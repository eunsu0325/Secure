#!/usr/bin/env bash
# Exp1 D6 variant — BJTU-V2 (ROI) end-to-end pipeline.
#
# Differences vs Tongji:
#   * Sessioned by phase (F enroll, S query) instead of session1/session2.
#   * t_max = 76 future palms; default --t_step 5 → schedule includes 76
#     explicitly via the build_t_schedule t_max-append rule.
#   * Builder reads `train/` (F) + `test/` (S) folders; pre-defined
#     dataset-author split files are detected and recorded as NOT USED.
#   * sessioned_K_pool=5 (BJTU has ~5 F-phase imgs/palm vs Tongji's 10).
#
# Usage:
#   BJTU_ROOT=/path/to/BJTU_PalmV2_ROI CKPT=/path/to/bjtu_mfn.pt \
#     bash experiments/run_d6_bjtu.sh

set -euo pipefail

BJTU_ROOT="${BJTU_ROOT:?set BJTU_ROOT to your BJTU_PalmV2(ROI) directory}"
CKPT="${CKPT:?set CKPT to a trained BJTU MFN checkpoint}"
OUT_DIR="${OUT_DIR:-experiments/generated/bjtu_d6}"
SPLIT_SEED="${SPLIT_SEED:-42}"
ORDER_SEED="${ORDER_SEED:-0}"
ENROLL_SEED="${ENROLL_SEED:-0}"
K="${K:-3}"
TARGET_FPIR="${TARGET_FPIR:-0.05}"
T_STEP="${T_STEP:-5}"
DEVICE="${DEVICE:-}"

mkdir -p "${OUT_DIR}"
EMB_PATH="${OUT_DIR}/mfn_bjtu_112_embeddings.npz"
RESULTS_DIR="${OUT_DIR}/protocols"

echo "1/3  Build BJTU manifest"
python -m exp1_baselines.datasets.bjtu_manifest \
    --root "${BJTU_ROOT}" --out "${OUT_DIR}" \
    --seed "${SPLIT_SEED}" --K "${K}" --enroll-seed "${ENROLL_SEED}"

echo "2/3  Extract embeddings"
EXTRACT_ARGS=(
    --model mfn --checkpoint "${CKPT}"
    --manifest_dir "${OUT_DIR}" --output_npz "${EMB_PATH}"
    --image_size 112
)
[ -n "${DEVICE}" ] && EXTRACT_ARGS+=(--device "${DEVICE}")
python exp1_baselines/extract_embeddings.py "${EXTRACT_ARGS[@]}"

echo "3/3  Run protocols (phase-sessioned)"
python -m exp1_baselines.eval.orchestrator \
    --embeddings  "${EMB_PATH}" \
    --manifest    "${OUT_DIR}/manifest.csv" \
    --out         "${RESULTS_DIR}" \
    --target_fpir "${TARGET_FPIR}" \
    --t_step      "${T_STEP}" \
    --order_seed  "${ORDER_SEED}" \
    --enroll_seed "${ENROLL_SEED}" \
    --K           "${K}" \
    --enroll_session none --enroll_phase F \
    --query_session  none --query_phase  S \
    --sessioned_K_pool 5

echo "BJTU D6 complete: results in ${RESULTS_DIR}"
