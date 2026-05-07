#!/usr/bin/env bash
# Exp1 D6 (BJTU-V2) — complete pipeline: build manifest, train MFN, extract,
# evaluate. Variant of run_d6_bjtu.sh that trains the backbone instead of
# assuming a pre-existing checkpoint.
#
# Usage:
#   BJTU_ROOT="$HOME/BJTU_extracted/BJTU_PalmV2(ROI)" \
#     bash experiments/run_d6_bjtu_full.sh

set -euo pipefail

BJTU_ROOT="${BJTU_ROOT:?set BJTU_ROOT to your BJTU_PalmV2(ROI) directory}"
OUT_DIR="${OUT_DIR:-experiments/generated/bjtu_full}"
SPLIT_SEED="${SPLIT_SEED:-42}"
ORDER_SEED="${ORDER_SEED:-0}"
ENROLL_SEED="${ENROLL_SEED:-0}"
K="${K:-3}"
TARGET_FPIR="${TARGET_FPIR:-0.05}"
T_STEP="${T_STEP:-5}"
DEVICE="${DEVICE:-mps}"
AMP="${AMP:-false}"
EPOCHS="${EPOCHS:-50}"

mkdir -p "${OUT_DIR}"
CKPT="${OUT_DIR}/mfn_arcface_bjtu_scratch_112.pt"
EMB="${OUT_DIR}/mfn_bjtu_112_embeddings.npz"
RESULTS="${OUT_DIR}/protocols"

echo "===== 1/4  Build BJTU manifest ====="
python -m exp1_baselines.datasets.bjtu_manifest \
    --root "${BJTU_ROOT}" --out "${OUT_DIR}" \
    --seed "${SPLIT_SEED}" --K "${K}" --enroll_seed "${ENROLL_SEED}"

echo
echo "===== 2/4  Train MFN+ArcFace 112 ====="
python exp1_baselines/train_arcface.py \
    --config exp1_baselines/configs/mfn_arcface_tongji_112.yaml \
    --manifest_dir "${OUT_DIR}" \
    --output       "${CKPT}" \
    --amp          "${AMP}" \
    --epochs       "${EPOCHS}" \
    --device       "${DEVICE}"

echo
echo "===== 3/4  Extract embeddings ====="
python exp1_baselines/extract_embeddings.py \
    --model        mfn \
    --checkpoint   "${CKPT}" \
    --manifest_dir "${OUT_DIR}" \
    --output_npz   "${EMB}" \
    --image_size   112 \
    --device       "${DEVICE}"

echo
echo "===== 4/4  Run protocols (sessioned by phase F/S) ====="
python -m exp1_baselines.eval.orchestrator \
    --embeddings  "${EMB}" \
    --manifest    "${OUT_DIR}/manifest.csv" \
    --out         "${RESULTS}" \
    --target_fpir "${TARGET_FPIR}" \
    --t_step      "${T_STEP}" \
    --order_seed  "${ORDER_SEED}" \
    --enroll_seed "${ENROLL_SEED}" \
    --K           "${K}" \
    --enroll_session none --enroll_phase F \
    --query_session  none --query_phase  S \
    --sessioned_K_pool 5

echo
echo "BJTU D6 complete: ${RESULTS}"
