#!/usr/bin/env bash
#SBATCH --job-name=greed-50k-report
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
vascular-ged analyze-50k \
  --experiment-root "${GREED_50K_ROOT:?}/checkpoints/greed_3d_50k" \
  --baseline-root "$GED_COMPUTATIONS_ROOT/checkpoints/greed_3d_learning_curve" \
  --report-root "$GREED_50K_ROOT/reports/greed_3d_50k"
