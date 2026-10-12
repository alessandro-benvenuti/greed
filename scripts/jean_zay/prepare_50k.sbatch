#!/usr/bin/env bash
#SBATCH --job-name=greed-50k-data
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
vascular-ged prepare-50k --source-root "$GED_COMPUTATIONS_ROOT" \
  --output-root "${GREED_50K_ROOT:?set GREED_50K_ROOT to a separate experiment directory}"
