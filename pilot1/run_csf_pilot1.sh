#!/usr/bin/env bash
# Pilot 1 CSF scoring sweep: fit + eval for the 20 intervention models,
# plain projections only (confirmatory pass; see MANIFEST.md section 5 and
# the projection decision of 2026-08-19). Run from code/ inside the frozen
# container; requires EXPERIMENT_ROOT_DIR and DATASET_ROOT_DIR.
#
# Usage:
#   bash pilot1/run_csf_pilot1.sh [filter]
# [filter] is an exp-name substring to split streams across GPUs, e.g.
#   CUDA_VISIBLE_DEVICES=0 nohup bash pilot1/run_csf_pilot1.sh run1 > csf_seed1.log 2>&1 &
# Env overrides: CSF_BATCH_SIZE, CSF_NUM_WORKERS (as in csf_fit.py),
# MODES (space-separated test modes; default = the cifar100 paper suite).
set -uo pipefail

GROUP=cifar100_intervention
FILTER="${1:-}"
MODES="${MODES:-iid_test ood_sncs_c10 ood_nsncs_svhn ood_nsncs_ti \
ood_nsncs_lsun_cropped ood_nsncs_lsun_resize ood_nsncs_isun \
ood_nsncs_textures ood_nsncs_places365}"
FLAGS="--no-rank_weight --no-rank_feature --ash None --use_cuda \
--temperature_scale --projections plain"

FAILED=()

run_model() {
    local name="$1"
    if [[ -n "${FILTER}" && "${name}" != *"${FILTER}"* ]]; then
        return 0
    fi
    local path="${GROUP}/${name}"
    local log="pilot1_csf_${name}.log"
    echo "=== FIT ${path} ==="
    if ! python csf_fit.py --model_path="${path}" ${FLAGS} \
            >> "${log}" 2>&1; then
        echo "!!! FIT FAILED: ${path} (see ${log})"
        FAILED+=("fit:${name}")
        return 0
    fi
    for mode in ${MODES}; do
        echo "=== EVAL ${path} [${mode}] ==="
        if ! python csf_eval.py --model_path="${path}" ${FLAGS} \
                --test_mode "${mode}" >> "${log}" 2>&1; then
            echo "!!! EVAL FAILED: ${path} [${mode}] (see ${log})"
            FAILED+=("eval:${name}:${mode}")
        fi
    done
}

for R in 1 2 3 4; do
    run_model "etfreg_bbvgg13_do0_run${R}_lam0.0"
    run_model "etfreg_bbvgg13_do0_run${R}_lam-0.1"
    run_model "etfreg_bbvgg13_do0_run${R}_lam0.3"
    run_model "etfreg_bbvgg13_do0_run${R}_lam1.0"
    run_model "etfhard_bbvgg13_do0_run${R}_lamhard"
done

echo "sweep complete"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf 'FAILURES (%d):\n' "${#FAILED[@]}"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
