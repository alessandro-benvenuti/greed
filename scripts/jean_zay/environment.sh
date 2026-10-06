#!/usr/bin/env bash
set -euo pipefail
module purge
module load gcc/11.5.0
module load git/2.53.0
module load cmake/3.31.4
module load gurobi/13.0.1
export SYNTHETIC_MRI_DATASET=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary
export GED_COMPUTATIONS_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/GED_computations
export GREED_REPOSITORY="$WORK/projects/greed"
