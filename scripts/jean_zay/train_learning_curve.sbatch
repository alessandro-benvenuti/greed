#!/usr/bin/env bash
#SBATCH --job-name=greed-curve
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-11%4
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
source "${GREED_REPOSITORY:?GREED_REPOSITORY must point to the GREED checkout}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"

counts=(1000 2000 4000 8000)
seeds=(161803 271828 314159)
count_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
count=${counts[$count_index]}
seed=${seeds[$seed_index]}
output="$GED_COMPUTATIONS_ROOT/checkpoints/greed_3d_learning_curve/pairs_${count}/seed_${seed}"

vascular-ged train \
  --config configs/greed_training_config.yaml \
  --train-count "$count" \
  --seed "$seed" \
  --output-dir "$output"
