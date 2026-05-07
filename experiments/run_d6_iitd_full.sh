#!/usr/bin/env bash
# Exp1 D6 (IITD) — complete pipeline: build manifest, train MFN, extract,
# evaluate. Differs from `run_d6_iitd.sh` in that it trains the backbone
# rather than assuming a pre-existing checkpoint. Use this when reproducing
# paper-scale results from scratch on IITD.
#
# Usage:
#   IITD_ROOT="$HOME/IITD_extracted/IITD Palmprint V1" \
#     bash experiments/run_d6_iitd_full.sh

set -euo pipefail

IITD_ROOT="${IITD_ROOT:?set IITD_ROOT to your IITD V1 root}"
OUT_DIR="${OUT_DIR:-experiments/generated/iitd_full}"
SPLIT_SEED="${SPLIT_SEED:-42}"
ORDER_SEED="${ORDER_SEED:-0}"
ENROLL_SEED="${ENROLL_SEED:-0}"
K="${K:-3}"
TARGET_FPIR="${TARGET_FPIR:-0.05}"
T_STEP="${T_STEP:-10}"
DEVICE="${DEVICE:-mps}"
AMP="${AMP:-false}"
EPOCHS="${EPOCHS:-50}"

mkdir -p "${OUT_DIR}"
CKPT="${OUT_DIR}/mfn_arcface_iitd_scratch_112.pt"
EMB="${OUT_DIR}/mfn_iitd_112_embeddings.npz"
RESULTS="${OUT_DIR}/protocols"

echo "===== 1/4  Build IITD manifest ====="
python -m exp1_baselines.datasets.iitd_manifest \
    --root "${IITD_ROOT}" --out "${OUT_DIR}" --seed "${SPLIT_SEED}"

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
echo "===== 4/4  Run protocols (non-sessioned) ====="
python -m exp1_baselines.eval.orchestrator \
    --embeddings  "${EMB}" \
    --manifest    "${OUT_DIR}/manifest.csv" \
    --out         "${RESULTS}" \
    --target_fpir "${TARGET_FPIR}" \
    --t_step      "${T_STEP}" \
    --order_seed  "${ORDER_SEED}" \
    --enroll_seed "${ENROLL_SEED}" \
    --K           "${K}" \
    --enroll_session none \
    --query_session  none \
    --non_sessioned

echo
echo "IITD D6 complete: ${RESULTS}"
