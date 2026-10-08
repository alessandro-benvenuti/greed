#!/usr/bin/env bash
#SBATCH --job-name=greed-ged
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=02:15:00
#SBATCH --array=0-399%8
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
source "${GREED_REPOSITORY:?GREED_REPOSITORY must point to the GREED checkout}/scripts/jean_zay/environment.sh"
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
if python scripts/jean_zay/shard_complete.py "$output" \
    --expected-rows "$chunk_size" --cost-config-id vascular_3d_raw_v1 --solver f2 --timeout 60; then
  exit 0
fi
python -c 'import pyged; print(f"Using pyged from {pyged.__file__}")'
vascular-ged run-shard --manifest "$manifest" --config configs/cost_config.yaml \
  --start "$local_start" --stop "$local_stop" --method f2 --timeout 60 --output "$output" \
  --reuse-dir "$GED_COMPUTATIONS_ROOT/results/pilot/shards"
