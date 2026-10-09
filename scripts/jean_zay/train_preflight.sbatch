#!/usr/bin/env bash
#SBATCH --job-name=greed-train-check
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?GREED_REPOSITORY must point to the GREED checkout}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
vascular-ged train --config configs/greed_training_config.yaml --preflight-only
