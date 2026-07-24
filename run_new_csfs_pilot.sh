#!/bin/bash
# E-F rebuttal pilot: evaluate ASH, ReAct, Mahalanobis++, and NCI over a set
# of existing FD-Shifts checkpoints. Meant for the SECOND GPU of an
# interactive allocation while the X8 Pool A extraction uses the first:
#
#   CUDA_VISIBLE_DEVICES=1 EXPERIMENTS_FILE=pilot_experiments.txt \
#     nohup bash run_new_csfs_pilot.sh > new_csfs_pilot.log 2>&1 &
#
# MUST run in an environment with the full pipeline (the paper container via
# `singularity exec --nv`, or the paper env) -- NOT the minimal x8 venv,
# because csf_fit/csf_eval import fd-shifts and torch cu117.
#
# Scope via environment:
#   EXPERIMENTS       space-separated fd-shifts experiment names, or
#   EXPERIMENTS_FILE  file with one experiment name per line ('#' comments ok)
#   ARMS              default "newcsf ash react"
#   TEST_MODES        default: iid_test + the six nsncs OOD modes; append the
#                     cross-source modes configured for your source (e.g. the
#                     new-class study modes) to cover near-OOD
#   HEAD_CSFS         families rescored under ASH/ReAct (default head-side)
#   ASH_METHOD        default ash_s@65
#   REACT_PCT         default 90 (per-checkpoint threshold from validation)
#
# Outputs land in each experiment's analysis/ dir as
# stats_RW0_RF0_ASH<method>_<mode>.csv; the newcsf arm merges MahaPP and NCI
# rows into the existing ASHNone stats files. Aggregate afterwards with
# aggregate_new_csfs_pilot.py. Failures on individual checkpoints are logged
# and skipped so one bad model cannot kill the night.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
set -a; source .env 2>/dev/null || true; set +a

ARMS="${ARMS:-newcsf ash react}"
TEST_MODES="${TEST_MODES:-iid_test ood_nsncs_svhn ood_nsncs_ti ood_nsncs_lsun_cropped ood_nsncs_lsun_resize ood_nsncs_isun ood_nsncs_textures ood_nsncs_places365}"
HEAD_CSFS="${HEAD_CSFS:-MSR,MLS,Energy,PE,PCE,GE,GEN,REN,GradNorm}"
ASH_METHOD="${ASH_METHOD:-ash_s@65}"
REACT_PCT="${REACT_PCT:-90}"
COMMON=(--no-rank_weight --no-rank_feature --use_cuda --temperature_scale)

if [ -n "${EXPERIMENTS_FILE:-}" ]; then
    mapfile -t EXPS < <(grep -v '^\s*#' "$EXPERIMENTS_FILE" | grep -v '^\s*$')
elif [ -n "${EXPERIMENTS:-}" ]; then
    read -r -a EXPS <<< "$EXPERIMENTS"
else
    echo "ERROR: set EXPERIMENTS or EXPERIMENTS_FILE" >&2
    exit 1
fi

echo "=== E-F new-CSF pilot ==="
echo "start: $(date '+%F %T')  pid: $$  node: $(hostname)"
echo "gpu: ${CUDA_VISIBLE_DEVICES:-all} | arms: $ARMS | ${#EXPS[@]} experiments"

run_fit_eval() {
    local exp="$1" ash="$2" csfs="$3" tag="$4"
    echo "--- [$tag] fit: $exp  ($(date '+%T'))"
    if ! python csf_fit.py --model_path="$exp" "${COMMON[@]}" \
            --ash "$ash" --csfs "$csfs" --projections none; then
        echo "FAIL [$tag] fit: $exp" | tee -a pilot_failures.log
        return 1
    fi
    for mode in $TEST_MODES; do
        echo "--- [$tag] eval $mode: $exp  ($(date '+%T'))"
        if ! python csf_eval.py --model_path="$exp" "${COMMON[@]}" \
                --ash "$ash" --csfs "$csfs" --projections none \
                --test_mode "$mode"; then
            echo "FAIL [$tag] eval $mode: $exp" | tee -a pilot_failures.log
        fi
    done
}

for exp in "${EXPS[@]}"; do
    echo "=== experiment: $exp ==="
    for arm in $ARMS; do
        case "$arm" in
            newcsf)
                run_fit_eval "$exp" None "MahalanobisPP,NCI" newcsf || true
                ;;
            ash)
                run_fit_eval "$exp" "$ASH_METHOD" "$HEAD_CSFS" ash || true
                ;;
            react)
                t=$(python react_calibrate.py --model_path "$exp" \
                        --percentile "$REACT_PCT" --use_cuda | tail -1)
                if [ -z "$t" ]; then
                    echo "FAIL [react] calibrate: $exp" | tee -a pilot_failures.log
                    continue
                fi
                echo "--- [react] threshold for $exp: $t"
                run_fit_eval "$exp" "react@$t" "$HEAD_CSFS" react || true
                ;;
            *)
                echo "unknown arm: $arm" >&2
                ;;
        esac
    done
done

echo "=== done: $(date '+%F %T') ==="
[ -f pilot_failures.log ] && { echo "failures:"; cat pilot_failures.log; } \
    || echo "no failures logged"
