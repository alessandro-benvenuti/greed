#!/usr/bin/env bash
set -euo pipefail
module purge
module load gcc/11.5.0
module load git/2.53.0
module load python/3.11.5
module load boost/1.86.0
module load cmake/3.31.4
module load gurobi/13.0.1
export SYNTHETIC_MRI_DATASET=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary
export GED_COMPUTATIONS_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/GED_computations
export GREED_REPOSITORY="${GREED_REPOSITORY:-$WORK/projects/greed}"
export GREED_ENV="${GREED_ENV:-$WORK/environments/greed-3d-vascular}"
if [[ ! -x "$GREED_ENV/bin/python" ]]; then
  echo "GREED virtual environment is missing: $GREED_ENV" >&2
  return 1 2>/dev/null || exit 1
fi
# Do not source activate here: an inherited active venv can restore its saved
# pre-module PATH and hide the Git/CMake/Gurobi binaries loaded above.
export VIRTUAL_ENV="$GREED_ENV"
export PATH="$GREED_ENV/bin:$PATH"
unset PYTHONHOME
hash -r
export PYTHONPATH="$GREED_REPOSITORY/pyged/lib${PYTHONPATH:+:$PYTHONPATH}"
gedlib_library_path="$GREED_REPOSITORY/pyged/ext/gedlib/ext/nomad.3.8.1/lib:$GREED_REPOSITORY/pyged/ext/gedlib/ext/libsvm.3.22:$GREED_REPOSITORY/pyged/ext/gedlib/ext/fann.2.2.0/lib"
export LD_LIBRARY_PATH="${gedlib_library_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset gedlib_library_path
