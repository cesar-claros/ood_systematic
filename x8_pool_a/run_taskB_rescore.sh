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
OOD="${OOD:-imagenet_wild_carnivores}"          # default: the within-family set built from the OpenOOD ImageNet image list
TAG="$(echo "$OOD" | tr "," "_" | sed "s/=[^,]*//g")"
N_OOD="${N_OOD:-1000}"                         # 20 classes x 50 validation images

IMGLIST="$DATASET_ROOT_DIR/openood/data/benchmark_imglist/imagenet/test_imagenet.txt"
echo "[$(date)] task B rescore start; OOD=$OOD; n_ood=$N_OOD; image list: $IMGLIST"
[ -f "$IMGLIST" ] || echo "[warn] $IMGLIST not found; the wild-carnivore set will be skipped"
for RUN in B_lora_seed0 B_lora_seed1 B_full_seed0 B_full_seed1; do
  if [ -f "$OUT_ROOT/$RUN/outcomes_rescore_$TAG.csv" ]; then echo "[$(date)] $RUN already rescored, skipping"; continue; fi
  echo "[$(date)] $RUN start"
  python x8_pool_a/adaptation_trajectory.py --rescore "$OUT_ROOT/$RUN" --ood "$OOD" --n-ood "$N_OOD" --data-root data \
    --device cuda --amp --out unused > "rescore_${RUN}.log" 2>&1
  STATUS=$?
  if [ $STATUS -ne 0 ]; then echo "[$(date)] $RUN FAILED (exit $STATUS); see rescore_${RUN}.log"; else echo "[$(date)] $RUN done"; fi
done
echo "[$(date)] task B rescore finished"
