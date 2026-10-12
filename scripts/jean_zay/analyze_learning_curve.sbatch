#!/usr/bin/env bash
#SBATCH --job-name=greed-curve-report
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?GREED_REPOSITORY must point to the GREED checkout}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"

vascular-ged analyze-learning-curve \
  --experiment-root "$GED_COMPUTATIONS_ROOT/checkpoints/greed_3d_learning_curve" \
  --report-root "$GED_COMPUTATIONS_ROOT/reports/greed_3d_learning_curve"
