#!/bin/bash
# Exploratory rescoring of the four finished task-B (Oxford-IIIT Pets) trajectories on near-semantic shifts.
# Declared roles: imagenet_wild_carnivores primary; ninco and ssb_hard secondary; inaturalist control. Exploratory, not confirmation.
#
# Run detached from <repo_path>/code inside the container, with .env sourced:
#   nohup bash x8_pool_a/run_taskB_rescore.sh > taskB_rescore.log 2>&1 &
#   tail -f taskB_rescore.log
#
# Reads each run's saved features and checkpoints under $EXPERIMENT_ROOT_DIR/adaptation_pilot/<run>/, writes
# outcomes_rescore_<tag>.csv and rescore_<tag>.json into the same directory; a run whose rescore CSV exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
: "${EXPERIMENT_ROOT_DIR:?set EXPERIMENT_ROOT_DIR (source .env)}"
: "${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR (source .env)}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

OUT_ROOT="$EXPERIMENT_ROOT_DIR/adaptation_pilot"
IMAGENET_VAL="${IMAGENET_VAL:-$DATASET_ROOT_DIR/openood/data/images_largescale/imagenet_1k/val}"
OOD="imagenet_wild_carnivores=$IMAGENET_VAL,ninco,ssb_hard,inaturalist"
TAG="imagenet_wild_carnivores_ninco_ssb_hard_inaturalist"

echo "[$(date)] task B rescore start; ImageNet val folder: $IMAGENET_VAL"
[ -d "$IMAGENET_VAL" ] || echo "[warn] $IMAGENET_VAL not found; the wild-carnivore set will be skipped (set IMAGENET_VAL=<path> to override)"
for RUN in B_lora_seed0 B_lora_seed1 B_full_seed0 B_full_seed1; do
  if [ -f "$OUT_ROOT/$RUN/outcomes_rescore_$TAG.csv" ]; then echo "[$(date)] $RUN already rescored, skipping"; continue; fi
  echo "[$(date)] $RUN start"
  python x8_pool_a/adaptation_trajectory.py --rescore "$OUT_ROOT/$RUN" --ood "$OOD" --n-ood 3669 --data-root data \
    --device cuda --amp --out unused > "rescore_${RUN}.log" 2>&1
  STATUS=$?
  if [ $STATUS -ne 0 ]; then echo "[$(date)] $RUN FAILED (exit $STATUS); see rescore_${RUN}.log"; else echo "[$(date)] $RUN done"; fi
done
echo "[$(date)] task B rescore finished"
