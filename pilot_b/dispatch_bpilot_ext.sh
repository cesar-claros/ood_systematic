#!/usr/bin/env bash
# B-axis dose-search DOWNWARD EXTENSION (protocol section 8 decision,
# 2026-08-21). Round 1 rejected every dose via GB3: even the weakest doses
# overshot the A1++ matching level (weakest varreg landed at vc 0.0887 vs
# target 0.1115) and flattened the within-class spectrum out of the
# reference span. Doses here target the matching region
# d_var_collapse in [-0.015, -0.007], from the measured dose-response
# (varreg ~ lam^0.4; ctrreg nearly flat in lam, hence the wider bracket).
# Search-space continuation: gates stay frozen (GB2 as amended). Committed
# prediction: logit_scale re-enters the span in this region;
# eig_max_over_mean decides matchability. If no dose qualifies here, the
# pilot ends with audit outcome C; no further extensions.
set -euo pipefail

GROUP=cifar100_intervention
COMMON="study=intervention data=cifar100_data exp.mode=train exp.group_name=${GROUP}"
FILTER="${1:-}"

run_one() {
    local name="$1"; shift
    if [[ -n "${FILTER}" && "${name}" != *"${FILTER}"* ]]; then
        return 0
    fi
    echo "=== ${name} ==="
    _fd_shifts_exec ${COMMON} exp.name="${name}" "$@" \
        2>&1 | tee "bpilot_${name}.log"
}

for R in 1 2; do
    SEED=$((1000 + R))
    for LAM in 0.003 0.01 0.03; do
        run_one "varreg_bbvgg13_do0_run${R}_lam${LAM}" \
            exp.global_seed=${SEED} \
            model.intervention_kind=varreg model.intervention_lam=${LAM}
    done
    for LAM in 0.00003 0.0001 0.0003; do
        run_one "ctrreg_bbvgg13_do0_run${R}_lam${LAM}" \
            exp.global_seed=${SEED} \
            model.intervention_kind=ctrreg model.intervention_lam=${LAM}
    done
done
echo "extension dispatch complete"
