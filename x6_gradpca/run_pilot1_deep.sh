#!/bin/bash
# X6-GradPCA Pilot 1: deep-gradient stage over the pre-declared roster.
#
#   EXPERIMENTS_FILE=pilot1_gradpca_experiments.txt \
#     nohup bash x6_gradpca/run_pilot1_deep.sh > gradpca_deep_pilot1.log 2>&1 &
#
# MUST run from the code/ repo root inside the paper container
# (singularity exec --nv ...): imports fd-shifts and torch cu117.
# Smoke first: SMOKE=2048 EXPERIMENTS="cifar100_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2" \
#     bash x6_gradpca/run_pilot1_deep.sh
# (2048-sample cap keeps every CIFAR-100 class populated in the fit split).
#
# Scope via environment:
#   EXPERIMENTS / EXPERIMENTS_FILE   as in run_new_csfs_pilot.sh
#   MODES     default: full CIFAR-100 mode roster (see deep_gradpca.py)
#   CHUNK     forward batch size override (default 128 CNN / 16 ViT)
#   SMOKE     cap samples per split (smoke test only; omit for the real run)
#   OUT_DIR   default x6_gradpca/outputs

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
set -a; source .env 2>/dev/null || true; set +a

OUT_DIR="${OUT_DIR:-x6_gradpca/outputs}"
EXTRA=()
[ -n "${MODES:-}" ] && EXTRA+=(--modes "$MODES")
[ -n "${CHUNK:-}" ] && EXTRA+=(--chunk "$CHUNK")
[ -n "${SMOKE:-}" ] && EXTRA+=(--smoke "$SMOKE")

if [ -n "${EXPERIMENTS_FILE:-}" ]; then
    mapfile -t EXPS < <(grep -v '^\s*#' "$EXPERIMENTS_FILE" | grep -v '^\s*$')
elif [ -n "${EXPERIMENTS:-}" ]; then
    read -r -a EXPS <<< "$EXPERIMENTS"
else
    echo "ERROR: set EXPERIMENTS or EXPERIMENTS_FILE" >&2
    exit 1
fi

echo "=== X6-GradPCA Pilot 1 deep stage ==="
echo "start: $(date '+%F %T')  node: $(hostname)  ${#EXPS[@]} experiments"
for exp in "${EXPS[@]}"; do
    echo "--- $exp  ($(date '+%T'))"
    if ! python x6_gradpca/deep_gradpca.py --model_path="$exp" --use_cuda \
            --out_dir "$OUT_DIR" "${EXTRA[@]}"; then
        echo "FAIL: $exp" | tee -a x6_gradpca/pilot1_failures.log
    fi
done
echo "=== done: $(date '+%F %T') ==="
[ -f x6_gradpca/pilot1_failures.log ] && { echo "failures:"; cat x6_gradpca/pilot1_failures.log; } \
    || echo "no failures logged"
