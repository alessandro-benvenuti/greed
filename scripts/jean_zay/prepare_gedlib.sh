#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd "$(dirname "$0")/../.." && pwd)
gedlib="$repository_root/pyged/ext/gedlib"
mkdir -p "$repository_root/pyged/ext"
if [[ ! -d "$gedlib/.git" ]]; then
  git clone --branch v1.0 --depth 1 https://github.com/dbblumenthal/gedlib "$gedlib"
fi
cd "$gedlib"
patch_file="$repository_root/pyged/patches/gedlib-v1-gurobi-bounds.patch"
if git apply --check "$patch_file" 2>/dev/null; then
  git apply "$patch_file"
elif ! git apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "GEDLIB bound patch is neither applicable nor already applied" >&2
  exit 1
fi
python install.py
if [[ ! -d ext/boost_1_82_0 ]]; then
  archive="$SLURM_TMPDIR/boost_1_82_0.tar.gz"
  curl -L --fail --retry 3 https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz -o "$archive"
  tar -xzf "$archive" -C ext
fi
