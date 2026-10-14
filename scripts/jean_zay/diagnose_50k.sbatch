#!/usr/bin/env bash
#SBATCH --job-name=greed-50k-check
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
root="${GREED_50K_ROOT:?}"
vascular-ged diagnose-50k \
  --experiment-root "$root/checkpoints/greed_3d_50k" \
  --validation-labels "$root/results/production/ged_labels_validation.csv" \
  --patient-split "$root/metadata/patient_split.csv" \
  --patch-index "$SYNTHETIC_MRI_DATASET/patch_index.csv" \
  --report-root "$root/reports/pre_100k_diagnostics"
