#!/usr/bin/env bash
#SBATCH --job-name=greed-data
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$GREED_REPOSITORY"
vascular-ged audit --output-root "$GED_COMPUTATIONS_ROOT"
vascular-ged split --output-root "$GED_COMPUTATIONS_ROOT" --seed 314159
vascular-ged sample --output-root "$GED_COMPUTATIONS_ROOT" --seed 271828
python scripts/jean_zay/record_provenance.py --stage audit_split_sampling \
  --output "$GED_COMPUTATIONS_ROOT/metadata/run_provenance.json"
