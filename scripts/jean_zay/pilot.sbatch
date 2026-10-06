#!/usr/bin/env bash
#SBATCH --job-name=greed-pilot
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-17%8
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$GREED_REPOSITORY"
shard_complete() { [[ -f "$1" ]] && [[ $(wc -l < "$1") -eq 2 ]]; }
for cost in raw sqrt3; do
  config=configs/cost_config.yaml
  [[ "$cost" == sqrt3 ]] && config=configs/cost_config_sqrt3.yaml
  for timeout in 10 60 300; do
    output="$GED_COMPUTATIONS_ROOT/results/pilot/shards/${cost}_f2_${timeout}_${SLURM_ARRAY_TASK_ID}.csv"
    shard_complete "$output" || vascular-ged run-shard --manifest "$GED_COMPUTATIONS_ROOT/manifests/pilot_pairs.csv" \
      --config "$config" --start "$SLURM_ARRAY_TASK_ID" --stop "$((SLURM_ARRAY_TASK_ID+1))" \
      --method f2 --timeout "$timeout" --output "$output"
  done
  output="$GED_COMPUTATIONS_ROOT/results/pilot/shards/${cost}_branch_${SLURM_ARRAY_TASK_ID}.csv"
  shard_complete "$output" || vascular-ged run-shard --manifest "$GED_COMPUTATIONS_ROOT/manifests/pilot_pairs.csv" \
    --config "$config" --start "$SLURM_ARRAY_TASK_ID" --stop "$((SLURM_ARRAY_TASK_ID+1))" \
    --method branch --timeout 0 --output "$output"
done
