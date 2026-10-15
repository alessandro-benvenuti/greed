#!/usr/bin/env bash
#SBATCH --job-name=greed-50k-extend
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
root="${GREED_50K_ROOT:?}"
seeds=(161803 271828)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
original="$root/checkpoints/greed_3d_50k/pairs_50000/seed_${seed}"
vascular-ged train --config "$original/resolved_config.yaml" \
  --epochs 200 --resume-from "$original" \
  --output-dir "$root/checkpoints/greed_3d_50k_extended/pairs_50000/seed_${seed}"
