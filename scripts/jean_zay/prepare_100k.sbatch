#!/usr/bin/env bash
#SBATCH --job-name=greed-100k-data
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
vascular-ged prepare-100k --source-root "${GREED_50K_ROOT:?}" \
  --output-root "${GREED_100K_ROOT:?}" --audit-root "$GED_COMPUTATIONS_ROOT"
