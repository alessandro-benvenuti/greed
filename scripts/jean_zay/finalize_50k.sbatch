#!/usr/bin/env bash
#SBATCH --job-name=greed-50k-merge
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
root="${GREED_50K_ROOT:?}"
production="$root/results/production"
vascular-ged merge --manifest "$root/manifests/greed_train_pairs.csv" \
  --shard-dir "$production/train_shards" --output "$production/ged_labels_train.csv" --expected-count 50000
vascular-ged combine --train "$production/ged_labels_train.csv" \
  --validation "$production/ged_labels_validation.csv" --output "$production/ged_labels_all.csv" \
  --expected-train 50000 --expected-validation 2000
vascular-ged label-summary --labels "$production/ged_labels_all.csv" \
  --output "$root/reports/label_dataset_summary.md"
