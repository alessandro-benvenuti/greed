#!/usr/bin/env bash
#SBATCH --job-name=greed-train
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$GREED_REPOSITORY"
vascular-ged train --config configs/greed_training_config.yaml
