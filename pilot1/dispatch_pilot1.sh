#!/usr/bin/env bash
# Pilot 1 dispatch: 15 paired trainings (code/pilot1/MANIFEST.md section 2).
# Run inside the FROZEN container (MANIFEST section 1) from any directory;
# requires EXPERIMENT_ROOT_DIR and DATASET_ROOT_DIR. Sequential by default;
# pass a subset filter as $1 (e.g. "run1" or "lam-0.1") to split across GPUs.
set -euo pipefail

GROUP=cifar100_intervention
COMMON="study=intervention data=cifar100_data exp.mode=train_test exp.group_name=${GROUP}"
FILTER="${1:-}"

run_one() {
    local name="$1"; shift
    if [[ -n "${FILTER}" && "${name}" != *"${FILTER}"* ]]; then
        return 0
    fi
    echo "=== ${name} ==="
    _fd_shifts_exec ${COMMON} exp.name="${name}" "$@" \
        2>&1 | tee "pilot1_${name}.log"
}

for R in 1 2 3; do
    SEED=$((1000 + R))
    run_one "etfreg_bbvgg13_do0_run${R}_lam0.0"  exp.global_seed=${SEED} \
        model.intervention_kind=etfreg model.intervention_lam=0.0
    run_one "etfreg_bbvgg13_do0_run${R}_lam-0.1" exp.global_seed=${SEED} \
        model.intervention_kind=etfreg model.intervention_lam=-0.1
    run_one "etfreg_bbvgg13_do0_run${R}_lam0.3"  exp.global_seed=${SEED} \
        model.intervention_kind=etfreg model.intervention_lam=0.3
    run_one "etfreg_bbvgg13_do0_run${R}_lam1.0"  exp.global_seed=${SEED} \
        model.intervention_kind=etfreg model.intervention_lam=1.0
    run_one "etfhard_bbvgg13_do0_run${R}_lamhard" exp.global_seed=${SEED} \
        model.intervention_mode=fixed_etf
done
echo "dispatch complete"
