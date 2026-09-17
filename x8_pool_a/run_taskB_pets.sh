#!/bin/bash
# Task B of the adaptation pilot: Oxford-IIIT Pets, four trajectories (LoRA and full fine-tuning, seeds 0 and 1),
# per Amendment 1 of documentation/adaptation_pilot_taskA_synthesis_2026-09-16.md. Same steps, batch and learning rates as task A.
#
# Run detached from <repo_path>/code inside the container, with .env sourced:
#   nohup bash x8_pool_a/run_taskB_pets.sh > taskB_pets.log 2>&1 &
#   tail -f taskB_pets.log
#
# Each trajectory writes to $EXPERIMENT_ROOT_DIR/adaptation_pilot/B_<method>_seed<s>/ and its own log next to this file's log.
# A trajectory whose outcomes.csv already exists is skipped, so the script resumes after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
: "${EXPERIMENT_ROOT_DIR:?set EXPERIMENT_ROOT_DIR (source .env)}"
: "${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR (source .env)}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

OUT_ROOT="$EXPERIMENT_ROOT_DIR/adaptation_pilot"
COMMON=(--backbone dinov2_vitb14 --data pets --data-root data --ood svhn,dtd,places365,food101
        --batch 64 --steps 3515 --checkpoints 0,350,700,1400,2100,3515
        --n-cls 37 --n-fit 2944 --n-val 736 --n-test 3669 --n-ood 3669 --device cuda --amp)

echo "[$(date)] task B start; out root $OUT_ROOT"
for SEED in 0 1; do
  for METHOD in lora full; do
    NAME="B_${METHOD}_seed${SEED}"
    if [ -f "$OUT_ROOT/$NAME/outcomes.csv" ]; then echo "[$(date)] $NAME already complete, skipping"; continue; fi
    if [ "$METHOD" = lora ]; then ARGS=(--method lora --lora-rank 8 --lr 2e-4); else ARGS=(--method full --lr 2e-5); fi
    echo "[$(date)] $NAME start"
    python x8_pool_a/adaptation_trajectory.py "${COMMON[@]}" "${ARGS[@]}" --seed "$SEED" --out "$OUT_ROOT/$NAME" > "traj_${NAME}.log" 2>&1
    STATUS=$?
    if [ $STATUS -ne 0 ]; then echo "[$(date)] $NAME FAILED (exit $STATUS); see traj_${NAME}.log"; else echo "[$(date)] $NAME done"; fi
  done
done
echo "[$(date)] task B finished"
