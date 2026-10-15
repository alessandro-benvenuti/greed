#!/usr/bin/env bash
#SBATCH --job-name=greed-test-inventory
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
vascular-ged test-patient-inventory \
  --patch-index "$SYNTHETIC_MRI_DATASET/patch_index.csv" \
  --patient-split "${GREED_50K_ROOT:?}/metadata/patient_split.csv" \
  --report-root "$GREED_50K_ROOT/reports/test_patient_provenance"
