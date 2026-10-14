#!/usr/bin/env bash
#SBATCH --job-name=greed-50k-train
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
source "${GREED_REPOSITORY:?}/scripts/jean_zay/environment.sh"
cd "$GREED_REPOSITORY"
root="${GREED_50K_ROOT:?}"
seeds=(161803 271828 314159)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
python - "$root" "$GED_COMPUTATIONS_ROOT" "$seed" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, original, seed = sys.argv[1:]
baseline = json.loads((Path(original) / f'checkpoints/greed_3d_learning_curve/pairs_8000/seed_{seed}/model_metadata.json').read_text())
for key, path in (
    ('validation_labels_sha256', Path(root) / 'results/production/ged_labels_validation.csv'),
    ('patient_split_sha256', Path(root) / 'metadata/patient_split.csv'),
    ('cost_config_sha256', Path('configs/cost_config.yaml')),
):
    if hashlib.sha256(path.read_bytes()).hexdigest() != baseline[key]:
        raise ValueError(f'50k training input differs from 8k baseline: {key}')
print('50k comparison inputs match the 8k baseline', flush=True)
PY
extra=()
if [[ "${GREED_TRAIN_PREFLIGHT:-0}" == 1 ]]; then
  extra+=(--preflight-only)
fi
vascular-ged train --config configs/greed_training_config.yaml \
  --train-count 50000 --seed "$seed" \
  --train-labels "$root/results/production/ged_labels_train.csv" \
  --validation-labels "$root/results/production/ged_labels_validation.csv" \
  --patient-split "$root/metadata/patient_split.csv" \
  --output-dir "$root/checkpoints/greed_3d_50k/pairs_50000/seed_${seed}" "${extra[@]}"
