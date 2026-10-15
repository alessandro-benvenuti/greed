#!/usr/bin/env bash
#SBATCH --job-name=greed-100k-ged
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=02:15:00
#SBATCH --array=0-999%8
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
root="${GREED_100K_ROOT:?}"
chunk_size=50
start=$((SLURM_ARRAY_TASK_ID * chunk_size))
stop=$((start + chunk_size))
output="$root/results/production/train_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
test -f "$root/metadata/expansion.json"
if python scripts/jean_zay/shard_complete.py "$output" \
    --expected-rows "$chunk_size" --cost-config-id vascular_3d_raw_v1 --solver f2 --timeout 60; then
  exit 0
fi
python -c 'import pyged; print(f"Using pyged from {pyged.__file__}")'
vascular-ged run-shard --manifest "$root/manifests/new_train_pairs.csv" \
  --config configs/cost_config.yaml --start "$start" --stop "$stop" \
  --method f2 --timeout 60 --output "$output"
python scripts/jean_zay/shard_complete.py "$output" \
  --expected-rows "$chunk_size" --cost-config-id vascular_3d_raw_v1 --solver f2 --timeout 60
