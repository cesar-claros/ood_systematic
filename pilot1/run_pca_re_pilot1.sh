#!/usr/bin/env bash
# Supplementary Pilot 1 pass: PCA_RecError (the registered E3 null "PCA_RE",
# MANIFEST.md section 5). The score is ProjectionFiltering's own
# reconstruction error and only exists as the global-projection variant, so
# the plain-only confirmatory sweep (run_csf_pilot1.sh) cannot produce it.
# This pass fits PF-global + Temperature-global only and MERGES the
# PCA_RecError_global row into the existing stats CSVs (csf_pipeline
# preserves other families' rows).
#
# Blinding note: PCA_RecError has no committed stage-2b direction cells (it
# is a pure E3 null) and its computation is deterministic given the frozen
# checkpoints, so running this pass after the plain sweep does not touch the
# committed predictions. The variant identification PCA_RE :=
# PCA_RecError_global (paper-roster canonical) is fixed in
# nc_csf_predictivity/interventions/outcome_analysis.py before any E3
# outcome is examined.
#
# Usage (same conventions as run_csf_pilot1.sh):
#   bash pilot1/run_pca_re_pilot1.sh [filter]
#   CUDA_VISIBLE_DEVICES=0 nohup bash pilot1/run_pca_re_pilot1.sh run1 > pcare_seed1.log 2>&1 &
set -uo pipefail

GROUP=cifar100_intervention
FILTER="${1:-}"
MODES="${MODES:-iid_test ood_sncs_c10 ood_nsncs_svhn ood_nsncs_ti \
ood_nsncs_lsun_cropped ood_nsncs_lsun_resize ood_nsncs_isun \
ood_nsncs_textures ood_nsncs_places365}"
FLAGS="--no-rank_weight --no-rank_feature --ash None --use_cuda \
--temperature_scale --projections global --csfs PCA_RecError"

FAILED=()

run_model() {
    local name="$1"
    if [[ -n "${FILTER}" && "${name}" != *"${FILTER}"* ]]; then
        return 0
    fi
    local path="${GROUP}/${name}"
    local log="pilot1_pcare_${name}.log"
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
