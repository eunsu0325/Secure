#!/usr/bin/env bash
# Exp1 D6 — Tongji end-to-end pipeline (real data, GPU required).
#
# Stages:
#   1. manifest      : tongji_manifest.py builds manifest.csv + metadata.json
#                      (file-integrity check requires all 12000 BMPs present)
#   2. train         : train_arcface.py trains MFN+ArcFace on train_backbone
#                      (~50 epochs, cosine annealing; batch_size 128 P32xK4)
#   3. extract       : extract_embeddings.py emits the canonical D1h NPZ
#                      (5600 eval imgs across 4 splits, 512-d L2-normalized)
#   4. evaluate      : eval/orchestrator.py runs Protocol A/B/C-fixed/C-recal
#                      and writes 4 protocol JSONs + meta.json with the
#                      B==C-recal endpoint sanity result
#
# Time / hardware:
#   * Stage 2 (train) is the expensive one — ~1-3 hours on a single
#     consumer GPU (e.g. RTX 3090) for 50 epochs. AMP (`--amp true`) cuts
#     time on Ampere+ GPUs at no accuracy cost.
#   * Stages 1 / 3 / 4 are CPU-only and finish in seconds-to-minutes.
#
# Usage:
#   TONGJI_ROOT=/path/to/Tongji_ROI bash experiments/run_d6_tongji.sh
#
# Override defaults via env vars:
#   TONGJI_ROOT       absolute path to Tongji_ROI/ (containing session1/ session2/)
#   OUT_DIR           where manifest, ckpt, embeddings, results land
#                     (default: experiments/generated/tongji_d6/)
#   SPLIT_SEED        manifest split RNG (default 42)
#   ORDER_SEED        Protocol C future-enrollment order (default 0)
#   ENROLL_SEED       per-palm K=3 selection (default 0 → [0,1,2])
#   K                 enrollment images per palm (default 3)
#   TARGET_FPIR       calibration target (default 0.05; also try 0.01)
#   T_STEP            Protocol C step size (default 10 for Tongji t_max=150)
#   AMP               "true" / "false" (default false; flip to true on Ampere+)
#   EPOCHS            override training epochs (default from yaml = 50)
#   DEVICE            "cuda" / "cpu" (default auto)

set -euo pipefail

TONGJI_ROOT="${TONGJI_ROOT:?set TONGJI_ROOT to your Tongji_ROI directory}"
OUT_DIR="${OUT_DIR:-experiments/generated/tongji_d6}"
SPLIT_SEED="${SPLIT_SEED:-42}"
ORDER_SEED="${ORDER_SEED:-0}"
ENROLL_SEED="${ENROLL_SEED:-0}"
K="${K:-3}"
TARGET_FPIR="${TARGET_FPIR:-0.05}"
T_STEP="${T_STEP:-10}"
AMP="${AMP:-false}"
EPOCHS="${EPOCHS:-}"
DEVICE="${DEVICE:-}"

mkdir -p "${OUT_DIR}"
MANIFEST_DIR="${OUT_DIR}"
CKPT_PATH="${OUT_DIR}/mfn_arcface_tongji_scratch_112.pt"
EMB_PATH="${OUT_DIR}/mfn_tongji_112_embeddings.npz"
RESULTS_DIR="${OUT_DIR}/protocols"

echo "==========================================="
echo " 1/4  Build Tongji manifest"
echo "==========================================="
python -m exp1_baselines.datasets.tongji_manifest \
    --root "${TONGJI_ROOT}" \
    --out  "${MANIFEST_DIR}" \
    --seed "${SPLIT_SEED}" \
    --K    "${K}" \
    --enroll-seed "${ENROLL_SEED}"

echo
echo "==========================================="
echo " 2/4  Train MFN+ArcFace at 112x112"
echo "==========================================="
TRAIN_ARGS=(
    --config exp1_baselines/configs/mfn_arcface_tongji_112.yaml
    --manifest_dir "${MANIFEST_DIR}"
    --output       "${CKPT_PATH}"
    --amp          "${AMP}"
)
[ -n "${EPOCHS}" ] && TRAIN_ARGS+=(--epochs "${EPOCHS}")
[ -n "${DEVICE}" ] && TRAIN_ARGS+=(--device "${DEVICE}")
python exp1_baselines/train_arcface.py "${TRAIN_ARGS[@]}"

echo
echo "==========================================="
echo " 3/4  Extract embeddings for eval splits"
echo "==========================================="
EXTRACT_ARGS=(
    --model        mfn
    --checkpoint   "${CKPT_PATH}"
    --manifest_dir "${MANIFEST_DIR}"
    --output_npz   "${EMB_PATH}"
    --image_size   112
)
[ -n "${DEVICE}" ] && EXTRACT_ARGS+=(--device "${DEVICE}")
python exp1_baselines/extract_embeddings.py "${EXTRACT_ARGS[@]}"

echo
echo "==========================================="
echo " 4/4  Run Protocols A/B/C-fixed/C-recal"
echo "==========================================="
python -m exp1_baselines.eval.orchestrator \
    --embeddings  "${EMB_PATH}" \
    --manifest    "${MANIFEST_DIR}/manifest.csv" \
    --out         "${RESULTS_DIR}" \
    --target_fpir "${TARGET_FPIR}" \
    --t_step      "${T_STEP}" \
    --order_seed  "${ORDER_SEED}" \
    --enroll_seed "${ENROLL_SEED}" \
    --K           "${K}" \
    --enroll_session session1 \
    --query_session  session2 \
    --sessioned_K_pool 10

echo
echo "==========================================="
echo " D6 complete"
echo "==========================================="
echo "outputs:"
echo "  manifest    : ${MANIFEST_DIR}/manifest.csv"
echo "  checkpoint  : ${CKPT_PATH}"
echo "  embeddings  : ${EMB_PATH}"
echo "  protocol JSON : ${RESULTS_DIR}/{protocol_a,protocol_b,protocol_c_fixed,protocol_c_recal,meta}.json"
echo
echo "next: inspect ${RESULTS_DIR}/meta.json — b_vs_c_recal_endpoint_sanity must be 'ok'"
