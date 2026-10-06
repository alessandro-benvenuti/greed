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
source "${GREED_REPOSITORY:?GREED_REPOSITORY must point to the GREED checkout}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
python -c 'import pyged; print(f"Using pyged from {pyged.__file__}")'
shard_complete() {
  python scripts/jean_zay/shard_complete.py "$1" \
    --expected-rows 1 --cost-config-id "$2" --solver "$3" --timeout "$4"
}
for cost in raw sqrt3; do
  config=configs/cost_config.yaml
  cost_config_id=vascular_3d_raw_v1
  if [[ "$cost" == sqrt3 ]]; then
    config=configs/cost_config_sqrt3.yaml
    cost_config_id=vascular_3d_sqrt3_v1
  fi
  for timeout in 10 60 300; do
    output="$GED_COMPUTATIONS_ROOT/results/pilot/shards/${cost}_f2_${timeout}_${SLURM_ARRAY_TASK_ID}.csv"
    shard_complete "$output" "$cost_config_id" f2 "$timeout" || vascular-ged run-shard --manifest "$GED_COMPUTATIONS_ROOT/manifests/pilot_pairs.csv" \
      --config "$config" --start "$SLURM_ARRAY_TASK_ID" --stop "$((SLURM_ARRAY_TASK_ID+1))" \
      --method f2 --timeout "$timeout" --output "$output"
  done
  output="$GED_COMPUTATIONS_ROOT/results/pilot/shards/${cost}_branch_${SLURM_ARRAY_TASK_ID}.csv"
  shard_complete "$output" "$cost_config_id" branch 0 || vascular-ged run-shard --manifest "$GED_COMPUTATIONS_ROOT/manifests/pilot_pairs.csv" \
    --config "$config" --start "$SLURM_ARRAY_TASK_ID" --stop "$((SLURM_ARRAY_TASK_ID+1))" \
    --method branch --timeout 0 --output "$output"
done
