#!/usr/bin/env bash
#SBATCH --job-name=greed-env
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
set -euo pipefail
source "${GREED_REPOSITORY:?GREED_REPOSITORY must point to the GREED checkout}/scripts/jean_zay/environment.sh"
mkdir -p "$GED_COMPUTATIONS_ROOT/metadata" "$GED_COMPUTATIONS_ROOT/logs"
cd "$GREED_REPOSITORY"
scripts/jean_zay/prepare_gedlib.sh
python scripts/jean_zay/record_environment.py > "$GED_COMPUTATIONS_ROOT/metadata/software_versions.json"
python scripts/jean_zay/gurobi_smoke.py
gurobi_c_library=$(find "$GUROBI_HOME/lib" -maxdepth 1 -name 'libgurobi[0-9]*.so' | sort -V | tail -1)
gurobi_link_name=$(basename "$gurobi_c_library" .so)
gurobi_link_name=${gurobi_link_name#lib}
g++ scripts/jean_zay/gurobi_smoke.cpp -std=c++17 -I"$GUROBI_HOME/include" -L"$GUROBI_HOME/lib" \
  -Wl,-rpath,"$GUROBI_HOME/lib" -lgurobi_c++ -l"$gurobi_link_name" -o "$SLURM_TMPDIR/gurobi_smoke"
"$SLURM_TMPDIR/gurobi_smoke"
pybind11_dir=$(python -m pybind11 --cmakedir)
cmake -S pyged -B pyged/build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$pybind11_dir"
cmake --build pyged/build --parallel 1
PYTHONPATH=pyged/lib python scripts/jean_zay/stock_f2_smoke.py
PYTHONPATH=pyged/lib pytest -q tests/test_graph_and_costs.py tests/test_pyged_integration.py
