#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: train_ce_local.sh SOURCE DROPOUT [MODE]

Train the five CE ResNet-18 runs for one source and dropout condition.

SOURCE   cifar10 | cifar100 | supercifar | tiny-imagenet-200
DROPOUT  0 (disabled) | 1 (enabled; p=0.1 for ResNet-18)
MODE     train | train_test (default: train_test)

Example (from code/):
  nohup ./train_ce_local.sh cifar100 1 train_test >/dev/null 2>&1 &
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 64
fi

source_dataset=$1
dropout=$2
mode=${3:-train_test}

case "$source_dataset" in
    cifar10 | cifar100 | supercifar | tiny-imagenet-200) ;;
    *)
        echo "Invalid source dataset: $source_dataset" >&2
        usage >&2
        exit 64
        ;;
esac

case "$dropout" in
    0 | 1) ;;
    *)
        echo "DROPOUT must be 0 or 1; received: $dropout" >&2
        exit 64
        ;;
esac

case "$mode" in
    train | train_test) ;;
    *)
        echo "MODE must be train or train_test; received: $mode" >&2
        exit 64
        ;;
esac

if ! command -v fd_shifts >/dev/null 2>&1; then
    echo "fd_shifts is not available in the active environment." >&2
    exit 127
fi

code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
log_dir=${CE_LOCAL_LOG_DIR:-"$code_dir/logs"}
mkdir -p "$log_dir"
log_dir=$(cd -- "$log_dir" && pwd)

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
master_log="$log_dir/ce_${source_dataset}_do${dropout}_${timestamp}.log"

cd "$code_dir"

for run_index in 0 1 2 3 4; do
    display_run=$((run_index + 1))
    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Starting run ${display_run}/5" \
        | tee -a "$master_log"
    if ! fd_shifts launch \
            --model=ce \
            --backbone=resnet18 \
            --dataset="$source_dataset" \
            --dropout="$dropout" \
            --run="$run_index" \
            --mode="$mode" 2>&1 | tee -a "$master_log"; then
        echo "Run ${display_run}/5 failed; stopping." \
            | tee -a "$master_log" >&2
        exit 1
    fi
    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Finished run ${display_run}/5" \
        | tee -a "$master_log"
done

echo "All five runs completed. Log: $master_log" | tee -a "$master_log"
