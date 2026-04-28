#!/usr/bin/env bash
# Exp1 appendix: XJTU / BMPD ROI / GPDS100Contactless × {CCNet, RepViT};
# Tongji RepViT-only.
#
# Each dataset is gated by an env var so you can run subsets:
#   RUN_XJTU=1, RUN_BMPD=1, RUN_GPDS100C=1, RUN_TONGJI=1.
#
# Path env vars (set before invoking):
#   XJTU_HUAWEI_ROOT, XJTU_IPHONE_ROOT (optional for cross_device)
#   BMPD_ROOT, GPDS100C_ROOT, TONGJI_ROOT
#
# RepViT runs require timm>=0.9. CCNet runs require checkpoints/<weights>.pth.

set -euo pipefail

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

build_if_needed() {
    local name="$1"; shift
    local builder="$1"; shift
    local need_ccnet="$1"; shift
    local need_repvit="$1"; shift
    local out_dir="experiments/generated/${name}"
    if [ "${FORCE_REBUILD}" = "1" ] || ! manifest_ready "${out_dir}" "${name}" "${need_ccnet}" "${need_repvit}"; then
        echo "[BUILD] ${name}"
        python "experiments/dataset_builders/${builder}" "$@" --out "${out_dir}" --seed "${SEED}"
    else
        echo "[SKIP-BUILD] ${out_dir} manifest ready"
    fi
}

run_yaml() {
    local yaml="$1"
    local extra_args=()
    if [ ! -f "${yaml}" ]; then
        echo "[ERROR] ${yaml} not found"
        return 1
    fi
    case "${yaml}" in
        *_ccnet.yaml)
            if [ -n "${PRETRAINED_PATH:-}" ]; then
                extra_args+=(--pretrained_path "${PRETRAINED_PATH}")
            fi
            ;;
        *_repvit.yaml)
            if [ -n "${REPVIT_CHECKPOINT_PATH:-}" ]; then
                extra_args+=(--checkpoint_path "${REPVIT_CHECKPOINT_PATH}")
            fi
            ;;
    esac
    echo "=============================="
    echo " Running ${yaml}"
    echo "=============================="
    python experiments/exp1_run.py --config "${yaml}" --strict_sanity "${extra_args[@]}"
}

# ----- XJTU -----
if [ "${RUN_XJTU:-0}" = "1" ]; then
    XJTU_HUAWEI_ROOT="${XJTU_HUAWEI_ROOT:-/path/to/XJTU/huawei}"
    XJTU_MODE="${XJTU_MODE:-flash_to_nature}"
    XJTU_NAME="xjtu_${XJTU_MODE}"
    if [ "${XJTU_MODE}" = "cross_device" ]; then
        if [ -z "${XJTU_IPHONE_ROOT:-}" ]; then
            echo "[ERROR] XJTU_MODE=cross_device requires XJTU_IPHONE_ROOT"
            exit 1
        fi
        build_if_needed "${XJTU_NAME}" build_xjtu.py 1 1 \
            --huawei_root "${XJTU_HUAWEI_ROOT}" --iphone_root "${XJTU_IPHONE_ROOT}" \
            --mode cross_device
    elif [ "${XJTU_MODE}" = "flash_to_nature" ]; then
        build_if_needed "${XJTU_NAME}" build_xjtu.py 1 1 \
            --huawei_root "${XJTU_HUAWEI_ROOT}" --mode flash_to_nature
    else
        echo "[ERROR] XJTU_MODE must be flash_to_nature or cross_device, got: ${XJTU_MODE}"
        exit 1
    fi
    run_yaml "experiments/generated/${XJTU_NAME}/exp1_${XJTU_NAME}_ccnet.yaml"
    run_yaml "experiments/generated/${XJTU_NAME}/exp1_${XJTU_NAME}_repvit.yaml"
fi

# ----- BMPD ROI -----
if [ "${RUN_BMPD:-0}" = "1" ]; then
    BMPD_ROOT="${BMPD_ROOT:-/path/to/BMPD_ROI}"
    build_if_needed bmpd_roi build_bmpd_roi.py 1 1 --root "${BMPD_ROOT}"
    run_yaml experiments/generated/bmpd_roi/exp1_bmpd_roi_ccnet.yaml
    run_yaml experiments/generated/bmpd_roi/exp1_bmpd_roi_repvit.yaml
fi

# ----- GPDS100 Contactless -----
if [ "${RUN_GPDS100C:-0}" = "1" ]; then
    GPDS100C_ROOT="${GPDS100C_ROOT:-/path/to/HandsGPDS100Contactless2bands}"
    build_if_needed gpds100_contactless build_gpds100_contactless.py 1 1 --root "${GPDS100C_ROOT}"
    run_yaml experiments/generated/gpds100_contactless/exp1_gpds100_contactless_ccnet.yaml
    run_yaml experiments/generated/gpds100_contactless/exp1_gpds100_contactless_repvit.yaml
fi

# ----- Tongji ROI (RepViT-only) -----
if [ "${RUN_TONGJI:-0}" = "1" ]; then
    TONGJI_ROOT="${TONGJI_ROOT:-/path/to/Tongji/ROI}"
    build_if_needed tongji_roi build_tongji_roi.py 0 1 --root "${TONGJI_ROOT}"
    run_yaml experiments/generated/tongji_roi/exp1_tongji_roi_repvit.yaml
fi

echo "=============================="
echo " run_exp1_appendix.sh complete"
echo "=============================="
