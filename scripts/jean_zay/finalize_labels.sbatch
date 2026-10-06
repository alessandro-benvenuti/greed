#!/usr/bin/env bash
#SBATCH --job-name=greed-merge
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$GREED_REPOSITORY"
production="$GED_COMPUTATIONS_ROOT/results/production"
vascular-ged merge --manifest "$GED_COMPUTATIONS_ROOT/manifests/greed_train_pairs.csv" \
  --shard-dir "$production/train_shards" --output "$production/ged_labels_train.csv" --expected-count 8000
vascular-ged merge --manifest "$GED_COMPUTATIONS_ROOT/manifests/greed_validation_pairs.csv" \
  --shard-dir "$production/validation_shards" --output "$production/ged_labels_validation.csv" --expected-count 2000
vascular-ged combine --train "$production/ged_labels_train.csv" \
  --validation "$production/ged_labels_validation.csv" --output "$production/ged_labels_all.csv"
vascular-ged label-summary --labels "$production/ged_labels_train.csv" "$production/ged_labels_validation.csv" "$production/ged_labels_all.csv" \
  --output "$GED_COMPUTATIONS_ROOT/reports/label_dataset_summary.md"
