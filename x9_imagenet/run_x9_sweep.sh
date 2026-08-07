#!/usr/bin/env bash
# x9 ImageNet-scale sweep runner (documentation/imagenet_scale_plan.md).
#
# Runs INSIDE the x9 container (open a shell in the sif first, e.g. via the
# tunnel or `singularity shell --nv x9_imagenet.sif`); wraps
# extract_and_score.py for a nohup launch. Per-model failure isolation and
# resume live in the driver (--skip-existing), so rerunning this script
# continues where it stopped.
#
# Env:
#   DATASET_ROOT_DIR  (required) data root: imagenet1k_raw/, imagenet1k_superset/,
#                     ood_parquet/, hf/ cache, x9_outputs/
#   MODELS            model tags or "all"           [default all]
#   FIT_SEED          superset draw seed (G3: rerun 2-3 models with 1)  [0]
#   HF_HOME           weights cache                 [default $DATASET_ROOT_DIR/hf]
#   EXTRA_ARGS        forwarded to extract_and_score.py (e.g. --no-kpca)
#
# Usage (inside the container, on a GPU node):
#   CUDA_VISIBLE_DEVICES=0 nohup bash x9_imagenet/run_x9_sweep.sh \
#       > x9_sweep.log 2>&1 &
set -euo pipefail

: "${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR}"
MODELS="${MODELS:-all}"
FIT_SEED="${FIT_SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export HF_HOME="${HF_HOME:-$DATASET_ROOT_DIR/hf}"
export PYTHONUNBUFFERED=1
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'" \
    || { echo "CUDA unavailable: run on a GPU node (and --nv if entering the sif manually)"; exit 1; }

echo "[x9 sweep] models=$MODELS fit_seed=$FIT_SEED hf_home=$HF_HOME"
python "$CODE_DIR/x9_imagenet/extract_and_score.py" \
    --data-root "$DATASET_ROOT_DIR" \
    --models $MODELS \
    --fit-seed "$FIT_SEED" \
    --skip-existing \
    $EXTRA_ARGS
echo "[x9 sweep] done"
