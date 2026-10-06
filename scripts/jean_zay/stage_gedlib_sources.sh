#!/usr/bin/env bash
# Network-only source staging for a Jean Zay login node. No compilation occurs here.
set -euo pipefail
repository_root=$(cd "$(dirname "$0")/../.." && pwd)
external_root="$repository_root/pyged/ext"
gedlib="$external_root/gedlib"
mkdir -p "$external_root"
if [[ -e "$gedlib" && ! -d "$gedlib/.git" ]]; then
  if ! rmdir "$gedlib" 2>/dev/null; then
    echo "$gedlib exists but is not a GEDLIB Git checkout; inspect it manually" >&2
    exit 1
  fi
fi
if [[ ! -d "$gedlib/.git" ]]; then
  git clone --branch v1.0 --depth 1 https://github.com/dbblumenthal/gedlib "$gedlib"
fi
patch_file="$repository_root/pyged/patches/gedlib-v1-gurobi-bounds.patch"
cd "$gedlib"
if git apply --check "$patch_file" 2>/dev/null; then
  git apply "$patch_file"
elif ! git apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "GEDLIB bound patch is neither applicable nor already applied" >&2
  exit 1
fi
if [[ ! -d ext/boost_1_82_0 ]]; then
  archive="$external_root/boost_1_82_0.tar.gz"
  curl -L --fail --retry 3 https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz -o "$archive"
  tar -xzf "$archive" -C ext
fi
printf 'Staged GEDLIB commit: '
git rev-parse HEAD
printf 'Boost directory: %s\n' "$gedlib/ext/boost_1_82_0"
