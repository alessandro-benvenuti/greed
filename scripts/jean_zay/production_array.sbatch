#!/usr/bin/env bash
#SBATCH --job-name=greed-ged
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=02:15:00
#SBATCH --array=0-399%32
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$GREED_REPOSITORY"
chunk_size=25
start=$((SLURM_ARRAY_TASK_ID * chunk_size))
stop=$((start + chunk_size))
if (( start < 8000 )); then
  manifest="$GED_COMPUTATIONS_ROOT/manifests/greed_train_pairs.csv"
  shard_dir="$GED_COMPUTATIONS_ROOT/results/production/train_shards"
  local_start=$start; local_stop=$stop
else
  manifest="$GED_COMPUTATIONS_ROOT/manifests/greed_validation_pairs.csv"
  shard_dir="$GED_COMPUTATIONS_ROOT/results/production/validation_shards"
  local_start=$((start - 8000)); local_stop=$((stop - 8000))
fi
mkdir -p "$shard_dir"
output="$shard_dir/shard_${SLURM_ARRAY_TASK_ID}.csv"
if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $((chunk_size + 1)) ]]; then
  exit 0
fi
vascular-ged run-shard --manifest "$manifest" --config configs/cost_config.yaml \
  --start "$local_start" --stop "$local_stop" --method f2 --timeout 300 --output "$output"
