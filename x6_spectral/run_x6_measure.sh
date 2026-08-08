#!/bin/bash
# X6 campaign stage 1: Tier-A spectral measurement over a manifest of
# checkpoints, sequential on one GPU, resumable (skips cells whose JSON
# already exists). Measurement only: no outcome tables are read, so this is
# safe to run before the rule freeze.
#
# Usage (from anywhere; inside the campaign container on the HPC):
#   bash x6_spectral/run_x6_measure.sh [manifest] [extra measure args...]
#   CSF_BATCH_SIZE=128 CSF_NUM_WORKERS=12 \
#     nohup bash x6_spectral/run_x6_measure.sh > x6_measure.log 2>&1 &
#
# Default manifest: manifest_dev_pool.txt (ConfidNet VGG13 + all ViT).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODE_DIR"
[ -f .env ] && source .env

MANIFEST="${1:-$SCRIPT_DIR/manifest_dev_pool.txt}"
shift || true
OUT="$SCRIPT_DIR/outputs"
mkdir -p "$OUT/logs"

total=0
done_n=0
fail_n=0
while IFS= read -r exp || [ -n "$exp" ]; do
    case "$exp" in ''|'#'*) continue ;; esac
    total=$((total + 1))
    slug="${exp//\//__}"
    if [ -f "$OUT/$slug.json" ]; then
        echo "[skip] $exp"
        continue
    fi
    echo "[run ] $exp"
    if python x6_spectral/measure_checkpoint.py --model_path="$exp" \
            --use_cuda --out_dir="$OUT" "$@" \
            > "$OUT/logs/$slug.log" 2>&1; then
        done_n=$((done_n + 1))
    else
        fail_n=$((fail_n + 1))
        echo "$exp" >> "$OUT/failures.log"
        echo "[FAIL] $exp (see logs/$slug.log)"
    fi
done < "$MANIFEST"

echo "manifest: $total cells; newly measured: $done_n; failures: $fail_n"
