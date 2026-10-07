# GREED: A Neural Framework for Learning Graph Distance Functions

This repository contains the official reference implementation for the paper ["GREED: A Neural Framework for Learning Graph Distance Functions"](https://openreview.net/pdf?id=3LBxVcnsEkV) accepted at NeurIPS 2022. `neuro` contains our implementation of the neural models presented in the paper along with supporting code for experiments. `pyged` contains our python wrapper over [GEDLIB](https://github.com/dbblumenthal/gedlib), which can be used to compute SED/GED values and graph alignments using non-neural techniques.

## Data and Model Weights

The data and trained models can be downloaded from this [Google Drive link](https://drive.google.com/file/d/1bRf6isnbfIrDc7V8xStlwEFX1ZtIMBEB/view?usp=sharing). Please see the README contained therein for further details.

If you have `gdown` installed (`pip install gdown` or `conda install -c conda-forge gdown`), you can download on terminal with:
```bash
gdown 1bRf6isnbfIrDc7V8xStlwEFX1ZtIMBEB
```

## Experiments

The Jupyter notebooks for the experiments in the paper can be found at the sister repository [greed-expts](https://github.com/rishabh-ranjan/greed-expts).

## Installation

We recommend using a `conda` environment for installation.

1. Install _Python_, _Jupyter_, _PyTorch_ and _PyTorch Geometric_ (also `networkx` and `matplotlib`). The code has been tested to work with _Python 3.6.13_, _PyTorch 1.8.0_ and _PyTorch Geometric 1.6.3_, but later versions are also expected to run smoothly.

2. Install _pyged_:

	2.1. Install [GEDLIB](https://dbblumenthal.github.io/gedlib/) at `pyged/ext/gedlib` as a header-only library (see Section 4.1 in the docs).

    Detailed steps are as follows:

    ```bash
    mkdir pyged/ext
    cd pyged/ext
    git clone --branch v1.0 --depth 1 https://github.com/dbblumenthal/gedlib
    cd gedlib
    python install.py
    ```

    On Jean Zay, use the provided `boost/1.86.0` module; do not unpack a
    Boost source tree into the project quota. On other systems, expose a
    compatible installation through `BOOST_ROOT`.

	2.2. Install [Gurobi 9.1.1](https://support.gurobi.com/hc/en-us/articles/360054352391-Gurobi-9-1-1-released) at `pyged/ext/gurobi911`. Later versions can be used with suitable naming changes. _Gurobi_ requires a licence. Free academic licenses are available. _Gurobi_ is required for ground truth SED computation. Alternatively, one could use one of the non-MIP methods available in _GEDLIB_ or use the generated data provided by us. To build without _Gurobi_, comment out `#define GUROBI` in `pyged/src/pyged.cpp`.

    Detailed steps are as follows:

    ```bash
    cd pyged/ext
    wget https://packages.gurobi.com/9.1/gurobi9.1.1_linux64.tar.gz
    tar -xzf gurobi9.1.1_linux64.tar.gz
    cd gurobi911/linux64/src/build
    make
    ```

	2.3. Install [PyBind11](https://pybind11.readthedocs.io/en/stable/installing.html#include-with-conda-forge).

    Example step:
    ```bash
    conda install -c conda-forge pybind11
    ```

	2.4. Build _pyged_ (you can install `cmake` with `conda` if it's not available):
	```bash
	mkdir pyged/build
	cd pyged/build
	cmake ..
	make
	```
	This will create a Python module for _pyged_ in `pyged/lib`.

## Usage

Check out the experiment notebooks at [greed-expts](https://github.com/rishabh-ranjan/greed-expts) for example usage. The notebooks contain code for training, testing, visualization, etc.

## Contact

If you face any difficulties in using this repo feel free to raise a GitHub issue (recommended) or reach out via email at rishabhranjan0207@gmail.com. I am unable to respond to queries sent to rishabh.ranjan.cs118@cse.iitd.ac.in in a timely manner.

## Normalized 3D vascular GED workflow

The `3d-vascular-geometric-ged` branch adds a reproducible path for SyntheticMRI
vascular graphs. It never reads RelationFormer validation/test graphs and does
not modify that repository. Points retain their stored canonical `[D,H,W]`
column order. The primary costs are node relabel `||p_i-p_j||_2`, node and edge
insertion/deletion `1`, and edge relabel `0`. Graphs are undirected and simple:
self-loops are dropped and duplicate/reverse-duplicate edges are collapsed.

```bash
export SYNTHETIC_MRI_DATASET=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary
export GED_COMPUTATIONS_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/GED_computations
vascular-ged audit --output-root "$GED_COMPUTATIONS_ROOT"
vascular-ged split --output-root "$GED_COMPUTATIONS_ROOT" --seed 314159
vascular-ged sample --output-root "$GED_COMPUTATIONS_ROOT" --seed 271828
```

Only `train/vtp` is eligible. Whole patients are assigned with a stable SHA-256
ordering before pairs are sampled. The sampler creates exactly 8,000 training
and 2,000 validation pairs separately, using corresponding-region,
log-size-matched, and unrestricted strategies over small/medium/large strata.
Paths are dataset-relative. Empty graphs are audited but excluded from this
first experiment; their analytic distance is `|V|+|E|`. A complete pairwise
matrix is never generated.

### Jean Zay

Clone this branch at `$WORK/projects/greed`, install a maintained environment
with `pip install -e '.[train]'`, and run all compilation/solver work via Slurm:

```bash
source scripts/jean_zay/environment.sh
sinfo -o '%P %l %a' | grep qos_cpu-t4
scripts/jean_zay/stage_gedlib_sources.sh  # login node: network staging only
sbatch scripts/jean_zay/verify_environment.sbatch
sbatch --dependency=afterok:<verify-job> scripts/jean_zay/audit_and_sample.sbatch
sbatch --dependency=afterok:<data-job> scripts/jean_zay/pilot.sbatch
```

The verification job records the environment, runs a real compute-node Gurobi
optimization, discovers Gurobi 13 through `GUROBI_HOME`, and builds/imports the
wrapper. The pilot runs 18 representative pairs with F2 and BRANCH at
10/60/300 seconds for raw and `sqrt(3)` scaling. Analyze it with:

```bash
vascular-ged analyze-pilot --shard-dir "$GED_COMPUTATIONS_ROOT/results/pilot/shards" --output-root "$GED_COMPUTATIONS_ROOT"
```

Do not submit production until the licence/build/stock-F2 smoke tests,
handcrafted tests, finite ordered pilot bounds, selected timeout ceiling (60 seconds),
CPU ceiling (166.67 hours), and concurrency checks all pass. Then record the
selected setup and submit the resumable 25-pair array:

```bash
sbatch scripts/jean_zay/production_array.sbatch
```

Each task uses one solver thread and one atomic shard. `vascular-ged merge`
rejects missing/extra/duplicate pairs, crossed patients, missing graphs,
non-finite or inverted bounds, malformed exact rows, and inconsistent configs.
The authoritative outputs are `ged_labels_train.csv`,
`ged_labels_validation.csv`, and their validated concatenation
`ged_labels_all.csv`. Use `vascular-ged label-summary` for the hash-bearing
summary. Generated artifacts, logs, environments, and checkpoints stay below
`GED_COMPUTATIONS_ROOT` and outside Git.

The validated pilot selected raw coordinate scaling, a 60-second F2 limit,
25-pair shards, and array concurrency 8. This matched the 300-second pilot's
exact fraction while materially reducing projected CPU use. Non-exact rows
retain their lower/upper interval, and compatible raw/60 pilot rows are reused.
For later exponential similarity experiments, the pilot selected
`lambda = 0.05`; this calibration does not change the GED labels.

### Model and later integration

The loader consumes continuous `[N,3]` features. The shared eight-layer GIN uses
64 hidden/embedding dimensions, sum pooling, and Euclidean embedding distance,
which guarantees symmetry, non-negativity, zero self-distance, and
pair-independent embeddings. Exact rows regress to their point label; bounded
rows use `ReLU(lower-prediction)^2 + ReLU(prediction-upper)^2`, never a midpoint.
Canonical unaugmented coordinates are used for labels and this initial training;
pair members are not independently rotated. Start the CPU pilot with
`sbatch scripts/jean_zay/train_cpu.sbatch`.

For later RelationFormer integration, load the learned encoder in the other
repository and place prediction and GT in the same augmented coordinate frame.
This branch intentionally keeps hard GT adjacency. Its encoder boundary can
later accept weighted message passing without changing the label definition;
soft-graph integration is not implemented here.

## Citation

```bibtex
@inproceedings{ranjan&al22,
  author = {Ranjan, Rishabh and Grover, Siddharth and Medya, Sourav and Chakaravarthy, Venkatesan and Sabharwal, Yogish and Ranu, Sayan},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {GREED: A Neural Framework for Learning Graph Distance Functions},
  booktitle = {Advances in Neural Information Processing Systems 36: Annual Conference
               on Neural Information Processing Systems 2022, NeurIPS 2022, November 29-Decemer 1, 2022},
  year = {2022},
}
```
