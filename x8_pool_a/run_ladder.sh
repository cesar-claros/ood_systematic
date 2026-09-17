#!/bin/bash
# H1 retention ladder (documentation/adaptation_pilot_next_hypotheses_2026-09-17.md): one seed per arm, two tasks.
# Arms: full fine-tuning at 2e-5, 5e-5, 1e-4, 2e-4; LoRA ranks 4, 8, 32 at 2e-4; partial (last 2 blocks) at 5e-5.
# Tasks: pets (shifts: imagenet_wild_carnivores, svhn, dtd) and imagenet200 (shifts: ninco, ssb_hard, inaturalist, dtd).
# Same steps, batch and checkpoints as the pilot. Kill criterion is in the note; this script only produces the trajectories.
#
#   nohup bash x8_pool_a/run_ladder.sh > ladder.log 2>&1 &      # all 16 trajectories, about 8 to 9 GPU-hours on a V100
#   TASKS=pets nohup bash x8_pool_a/run_ladder.sh > ladder_pets.log 2>&1 &   # one task
#   ARMS="full_1e-4 lora_r32" TASKS=pets bash x8_pool_a/run_ladder.sh          # selected arms
# Resumable: an arm whose outcomes.csv exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
: "${EXPERIMENT_ROOT_DIR:?set EXPERIMENT_ROOT_DIR (source .env)}"
: "${DATASET_ROOT_DIR:?set DATASET_ROOT_DIR (source .env)}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

OUT_ROOT="$EXPERIMENT_ROOT_DIR/adaptation_ladder"
TASKS="${TASKS:-pets imagenet200}"
ARMS="${ARMS:-full_2e-5 full_5e-5 full_1e-4 full_2e-4 lora_r4 lora_r8 lora_r32 partial_5e-5}"
SEED="${SEED:-0}"
IMGLIST_WILD="$DATASET_ROOT_DIR/openood/data/benchmark_imglist/imagenet/test_imagenet.txt"

task_args() {   # task-specific data, sizes and shifts
  case "$1" in
    pets)        echo "--data pets --n-cls 37 --n-fit 2944 --n-val 736 --n-test 3669 --n-ood 1000 --ood imagenet_wild_carnivores,svhn,dtd" ;;
    imagenet200) echo "--data imagenet200 --n-cls 200 --n-fit 45000 --n-val 5000 --n-test 9000 --n-ood 5000 --ood ninco,ssb_hard,inaturalist,dtd" ;;
    *) echo "unknown task $1" >&2; return 1 ;;
  esac
}
arm_args() {
  case "$1" in
    full_*)    echo "--method full --lr ${1#full_}" ;;
    lora_r*)   echo "--method lora --lora-rank ${1#lora_r} --lr 2e-4" ;;
    partial_*) echo "--method partial --train-last-blocks 2 --lr ${1#partial_}" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

echo "[$(date)] ladder start; tasks: $TASKS; arms: $ARMS; seed $SEED; out root $OUT_ROOT"
[ -f "$IMGLIST_WILD" ] || echo "[warn] $IMGLIST_WILD not found; the wild-carnivore shift will be skipped on pets"
for TASK in $TASKS; do
  TA=$(task_args "$TASK") || continue
  for ARM in $ARMS; do
    AA=$(arm_args "$ARM") || continue
    NAME="${TASK}_${ARM}_seed${SEED}"
    if [ -f "$OUT_ROOT/$NAME/outcomes.csv" ]; then echo "[$(date)] $NAME already complete, skipping"; continue; fi
    echo "[$(date)] $NAME start"
    python x8_pool_a/adaptation_trajectory.py --backbone dinov2_vitb14 --data-root data $TA $AA --batch 64 --seed "$SEED" \
      --steps 3515 --checkpoints 0,350,700,1400,2100,3515 --out "$OUT_ROOT/$NAME" --device cuda --amp > "traj_${NAME}.log" 2>&1
    STATUS=$?
    if [ $STATUS -ne 0 ]; then echo "[$(date)] $NAME FAILED (exit $STATUS); see traj_${NAME}.log"; else echo "[$(date)] $NAME done"; fi
  done
done
echo "[$(date)] ladder finished"
