#!/usr/bin/env bash
set -euo pipefail
GREED_GIT_EXECUTABLE=$(command -v git || true)
if [[ -z "$GREED_GIT_EXECUTABLE" ]]; then
  echo "git must be available before loading the GREED Jean Zay environment" >&2
  exit 1
fi
module purge
module load gcc/11.5.0
module load cmake/3.31.4
module load gurobi/13.0.1
export PATH="$(dirname "$GREED_GIT_EXECUTABLE"):$PATH"
export SYNTHETIC_MRI_DATASET=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary
export GED_COMPUTATIONS_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/GED_computations
export GREED_REPOSITORY="$WORK/projects/greed"
