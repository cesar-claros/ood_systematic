#!/bin/bash
#SBATCH --job-name=x8_pool_a
#SBATCH --partition=<GPU_PARTITION>        # e.g. the T4 or A100 partition used for the paper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=x8_pool_a_%j.log
# Optional: #SBATCH --account=<ACCOUNT>

# X8 Pool A feature extraction: frozen DINOv2 ViT-B/14 + CLIP ViT-B/16 over
# the 4 ID sources (train+test) and 6 OOD sets. Expected wall-clock 1-3 h on
# a single modern GPU; the 8 h limit is generous headroom.
#
# Prerequisites (see documentation/x8_pool_a_dispatch_runbook.md):
#   1. repo synced to the cluster, run from the repo's code/ directory
#   2. venv at $VENV with torch/torchvision/open-clip-torch/tinyimagenet/
#      loguru/tqdm/numpy/pillow
#   3. encoder weights pre-downloaded on the login node (torch.hub + open_clip
#      caches) if compute nodes lack internet
#   4. EXPERIMENT_ROOT_DIR and DATASET_ROOT_DIR exported (or .env sourced)

set -euo pipefail

VENV="${VENV:-$HOME/x8env}"
source "$VENV/bin/activate"
set -a; source .env 2>/dev/null || true; set +a
export EXPERIMENT_ROOT_DIR="${EXPERIMENT_ROOT_DIR:?set EXPERIMENT_ROOT_DIR}"
export DATASET_ROOT_DIR="${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR}"

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "EXPERIMENT_ROOT_DIR=$EXPERIMENT_ROOT_DIR"

for ENC in dinov2_vitb14 clip_vitb16; do
    echo "=== $ENC ==="
    python x8_pool_a/extract_features.py --encoder "$ENC" --dataset all \
        --batch-size 256 --num-workers "$SLURM_CPUS_PER_TASK"
done

echo "=== done; inventory ==="
ls -lh "$EXPERIMENT_ROOT_DIR/pool_a/features/"
