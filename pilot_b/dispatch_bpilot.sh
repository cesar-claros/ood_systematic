#!/usr/bin/env bash
# B-axis geometry-only dose search (documentation/B_axis_pilot_protocol.md).
# Two variability mechanisms, two paired seeds each, TRAIN ONLY: the
# protocol forbids computing any OOD or detector score during the search
# (exp.mode=train; geometry comes from extract_manipulation afterwards).
# Baselines are NOT retrained: the Pilot 1 etfreg lam0.0 runs for seeds
# 1001/1002 are the paired references.
#
# B1 = varreg (within/between scatter ratio, in the frozen Pilot 1 image
# at fork commit 9775fc3). B2 = ctrreg (center-loss contraction, needs the
# fork commit adding center_loss_penalty + a container rebuild; the
# protocol records the new commit/digest before B2 dispatch).
#
# Dose grids are search space, not registered quantities; the smoke rule
# (protocol section 4) allows rescaling a grid by decades based only on
# the logged train/penalty_term magnitude. Filter arg as in pilot1, e.g.:
#   bash pilot_b/dispatch_bpilot.sh varreg     # B1 only (frozen image)
#   bash pilot_b/dispatch_bpilot.sh ctrreg     # B2 only (rebuilt image)
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
    for LAM in 0.1 0.3 1.0 3.0; do
        run_one "varreg_bbvgg13_do0_run${R}_lam${LAM}" \
            exp.global_seed=${SEED} \
            model.intervention_kind=varreg model.intervention_lam=${LAM}
    done
    # Grid rescaled one decade up per the smoke rule (protocol section 8,
    # 2026-08-20): measured penalty ~45 at epoch 8, so these doses give
    # lam*penalty ~ 0.14-4.5, inside the [0.05, 5] band.
    for LAM in 0.003 0.01 0.03 0.1; do
        run_one "ctrreg_bbvgg13_do0_run${R}_lam${LAM}" \
            exp.global_seed=${SEED} \
            model.intervention_kind=ctrreg model.intervention_lam=${LAM}
    done
done
echo "dispatch complete"
