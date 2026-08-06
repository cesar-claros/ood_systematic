#!/usr/bin/env bash
# x9 ImageNet-scale sweep runner (documentation/imagenet_scale_plan.md).
#
# Wraps extract_and_score.py for a nohup launch on a GPU node inside the
# x9 container. Per-model failure isolation and resume live in the driver
# (--skip-existing), so rerunning this script continues where it stopped.
#
# Env:
#   DATASET_ROOT_DIR  (required) data root: imagenet1k_raw/, imagenet1k_superset/,
#                     ood_parquet/, hf/ cache, x9_outputs/
#   SIF               path to x9_imagenet.sif       [default ./x9_imagenet.sif]
#   MODELS            model tags or "all"           [default all]
#   FIT_SEED          superset draw seed (G3: rerun 2-3 models with 1)  [0]
#   EXTRA_ARGS        forwarded to extract_and_score.py (e.g. --no-kpca)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash x9_imagenet/run_x9_sweep.sh \
#       > x9_sweep.log 2>&1 &
set -euo pipefail

: "${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR}"
SIF="${SIF:-./x9_imagenet.sif}"
MODELS="${MODELS:-all}"
FIT_SEED="${FIT_SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

[ -f "$SIF" ] || { echo "sif not found: $SIF (set SIF=...)"; exit 1; }

echo "[x9 sweep] models=$MODELS fit_seed=$FIT_SEED sif=$SIF"
singularity exec --nv \
    --env HF_HOME="$DATASET_ROOT_DIR/hf" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" \
    python "$CODE_DIR/x9_imagenet/extract_and_score.py" \
        --data-root "$DATASET_ROOT_DIR" \
        --models $MODELS \
        --fit-seed "$FIT_SEED" \
        --skip-existing \
        $EXTRA_ARGS
echo "[x9 sweep] done"
