#!/bin/bash
# X8 Pool A feature extraction on an interactive node, meant for nohup:
#
#   venv path:
#     nohup bash x8_pool_a/run_pool_a_hpc.sh > x8_extract.log 2>&1 &
#   container path (inside the allocation):
#     nohup singularity exec --nv systematic_ood.sif \
#       bash x8_pool_a/run_pool_a_hpc.sh > x8_extract.log 2>&1 &
#
# Then: tail -f x8_extract.log
#
# nohup survives a dropped SSH session but not the end of the allocation, so
# the interactive job must outlive the run (expect 1-3 h on a single GPU).
# The extractor skips any {encoder}_{dataset}_{split}.npz that already exists,
# so rerunning after an interruption resumes where it stopped.
#
# Overridable environment:
#   VENV      venv to activate if present (default $HOME/x8env); skipped when
#             absent, e.g. inside the container where python already has deps
#   ENCODERS  space-separated list (default "dinov2_vitb14 clip_vitb16")
#   BATCH     extractor batch size (default 256)
#   WORKERS   dataloader workers (default SLURM_CPUS_PER_TASK or nproc, max 8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODE_DIR"

VENV="${VENV:-$HOME/x8env}"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

set -a; source .env 2>/dev/null || true; set +a
export EXPERIMENT_ROOT_DIR="${EXPERIMENT_ROOT_DIR:?set EXPERIMENT_ROOT_DIR (or provide .env)}"
export DATASET_ROOT_DIR="${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR (or provide .env)}"

ENCODERS="${ENCODERS:-dinov2_vitb14 clip_vitb16}"
BATCH="${BATCH:-256}"
default_workers="${SLURM_CPUS_PER_TASK:-$(nproc)}"
WORKERS="${WORKERS:-$(( default_workers < 8 ? default_workers : 8 ))}"

echo "=== X8 Pool A extraction ==="
echo "start: $(date '+%F %T')  pid: $$  node: $(hostname)"
echo "python: $(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi unavailable; extraction will fall back to CPU"
echo "EXPERIMENT_ROOT_DIR=$EXPERIMENT_ROOT_DIR"
echo "encoders: $ENCODERS  batch: $BATCH  workers: $WORKERS"

for ENC in $ENCODERS; do
    echo "=== $ENC  ($(date '+%F %T')) ==="
    python x8_pool_a/extract_features.py --encoder "$ENC" --dataset all \
        --batch-size "$BATCH" --num-workers "$WORKERS"
done

echo "=== done: $(date '+%F %T') ==="
ls -lh "$EXPERIMENT_ROOT_DIR/pool_a/features/"
