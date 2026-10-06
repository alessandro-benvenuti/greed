#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd "$(dirname "$0")/../.." && pwd)
gedlib="$repository_root/pyged/ext/gedlib"
if [[ ! -d "$gedlib/.git" ]]; then
  echo "GEDLIB sources are not staged. Run scripts/jean_zay/stage_gedlib_sources.sh on the login node first." >&2
  exit 1
fi
cd "$gedlib"
patch_file="$repository_root/pyged/patches/gedlib-v1-gurobi-bounds.patch"
if ! git apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "GEDLIB bound patch is not applied. Re-run source staging on the login node." >&2
  exit 1
fi
if [[ ! -d ext/boost_1_82_0 ]]; then
  echo "Boost sources are not staged. Re-run source staging on the login node." >&2
  exit 1
fi
python install.py
