#!/usr/bin/env bash
# Exp1 RepViT robustness: BJTU V2 / IITD Segmented / GPDS150hand × RepViT,
# all --strict_sanity. RepViT is the CCNet-artifact defence backbone.
#
# Prerequisites:
#   * pip install timm>=0.9
#   * Manifest already built (run experiments/run_exp1_main.sh first, or build
#     each dataset's manifest separately).
#   * Optional: REPVIT_CHECKPOINT_PATH env to use a local pinned weight file
#     instead of timm's pretrained download. Without it, timm downloads from
#     HuggingFace.
#
# Set RUN_IITD_REPVIT=1 to include IITD; default skips it because the IITD
# RepViT pass takes longer.

set -euo pipefail

BJTU_ROOT="${BJTU_ROOT:-/path/to/BJTU_PalmV2_ROI}"
IITD_ROOT="${IITD_ROOT:-/path/to/IITD/Segmented}"
GPDS150_ROOT="${GPDS150_ROOT:-/path/to/HandsGPDS150}"
SEED="${SEED:-42}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

manifest_ready() {
    local out_dir="$1"
    local name="$2"
    local need_ccnet="$3"
    local need_repvit="$4"

    [ -f "${out_dir}/all.txt" ] || return 1
    [ -f "${out_dir}/identity_split.json" ] || return 1
    [ -f "${out_dir}/sample_split.json" ] || return 1
    [ -f "${out_dir}/metadata.json" ] || return 1
    if [ "${need_ccnet}" = "1" ]; then
        [ -f "${out_dir}/exp1_${name}_ccnet.yaml" ] || return 1
    fi
    if [ "${need_repvit}" = "1" ]; then
        [ -f "${out_dir}/exp1_${name}_repvit.yaml" ] || return 1
    fi
}

build_dataset() {
    local name="$1"; shift
    local builder="$1"; shift
    local out_dir="experiments/generated/${name}"

    if [ "${FORCE_REBUILD}" = "1" ] || ! manifest_ready "${out_dir}" "${name}" 0 1; then
        echo "=============================="
        echo " Building manifest: ${name}"
        echo "=============================="
        python "experiments/dataset_builders/${builder}" \
            "$@" --out "${out_dir}" --seed "${SEED}"
    else
        echo "[SKIP-BUILD] ${out_dir} manifest ready"
    fi
}

run_repvit() {
    local name="$1"
    local out_dir="experiments/generated/${name}"
    local extra_args=()
    if [ -n "${REPVIT_CHECKPOINT_PATH:-}" ]; then
        extra_args+=(--checkpoint_path "${REPVIT_CHECKPOINT_PATH}")
    fi

    echo "=============================="
    echo " Running RepViT on ${name}"
    echo "=============================="
    python experiments/exp1_run.py \
        --config "${out_dir}/exp1_${name}_repvit.yaml" \
        --strict_sanity \
        "${extra_args[@]}"
}

build_dataset bjtu_v2 build_bjtu_roi.py --root "${BJTU_ROOT}"
run_repvit bjtu_v2

build_dataset gpds150hand build_gpds150hand.py --root "${GPDS150_ROOT}"
run_repvit gpds150hand

if [ "${RUN_IITD_REPVIT:-0}" = "1" ]; then
    build_dataset iitd_segmented build_iitd_segmented.py --root "${IITD_ROOT}"
    run_repvit iitd_segmented
else
    echo "[SKIP] iitd_segmented RepViT (set RUN_IITD_REPVIT=1 to include)"
fi

echo "=============================="
echo " run_exp1_repvit.sh complete"
echo "=============================="
