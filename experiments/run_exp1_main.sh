#!/usr/bin/env bash
# Exp1 main: BJTU V2 / IITD Segmented / GPDS150hand × CCNet, all --strict_sanity.
#
# Prerequisites (per dataset):
#   * The dataset images on disk under the path you pass to the builder.
#   * checkpoints/tongji.pth (or override via PRETRAINED_PATH env).
#   * experiments/generated/<dataset>/all.txt + identity_split.json + sample_split.json
#     produced by the matching builder.
#
# Usage:
#   bash experiments/run_exp1_main.sh
#
# Override paths by setting env vars before invoking, e.g.:
#   BJTU_ROOT=/data/BJTU_PalmV2_ROI IITD_ROOT=/data/IITD/Segmented \
#     GPDS150_ROOT=/data/HandsGPDS150 bash experiments/run_exp1_main.sh

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

    if [ "${FORCE_REBUILD}" = "1" ] || ! manifest_ready "${out_dir}" "${name}" 1 0; then
        echo "=============================="
        echo " Building manifest: ${name}"
        echo "=============================="
        python "experiments/dataset_builders/${builder}" \
            "$@" --out "${out_dir}" --seed "${SEED}"
    else
        echo "[SKIP-BUILD] ${out_dir} manifest ready"
    fi
}

run_ccnet() {
    local name="$1"
    local out_dir="experiments/generated/${name}"
    local extra_args=()
    if [ -n "${PRETRAINED_PATH:-}" ]; then
        extra_args+=(--pretrained_path "${PRETRAINED_PATH}")
    fi

    echo "=============================="
    echo " Running CCNet on ${name}"
    echo "=============================="
    python experiments/exp1_run.py \
        --config "${out_dir}/exp1_${name}_ccnet.yaml" \
        --strict_sanity \
        "${extra_args[@]}"
}

build_dataset bjtu_v2 build_bjtu_roi.py --root "${BJTU_ROOT}"
run_ccnet bjtu_v2

build_dataset iitd_segmented build_iitd_segmented.py --root "${IITD_ROOT}"
run_ccnet iitd_segmented

build_dataset gpds150hand build_gpds150hand.py --root "${GPDS150_ROOT}"
run_ccnet gpds150hand

echo "=============================="
echo " run_exp1_main.sh complete"
echo "=============================="
