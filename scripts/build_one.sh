#!/usr/bin/env bash
set -euo pipefail

num="${1:?problem number (e.g. 03) required}"
out="${2:?output binary path required}"
src_prefix="${3:-mine}" # "mine" (default) or "gold"

midterm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$(dirname "$out")"

shopt -s nullglob
cu_files=("$midterm_dir/solution/${src_prefix}/${num}_"*.cu)
cpp_files=("$midterm_dir/solution/${src_prefix}/${num}_"*.cpp)
shopt -u nullglob

src=""
if ((${#cu_files[@]})); then
  src="${cu_files[0]}"
elif ((${#cpp_files[@]})); then
  src="${cpp_files[0]}"
else
  echo "No solution file found for problem $num in solution/${src_prefix}"
  exit 2
fi

if [[ "$src" == *.cu ]]; then
  nvcc_bin="${NVCC:-}"
  if [[ -z "$nvcc_bin" ]]; then
    nvcc_bin="$(command -v nvcc 2>/dev/null || true)"
  fi
  if [[ -z "$nvcc_bin" ]]; then
    for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
      if [[ -x "$p" ]]; then
        nvcc_bin="$p"
        break
      fi
    done
  fi

  if [[ -z "$nvcc_bin" ]]; then
    echo "SKIP: CUDA source ($src) but nvcc not found"
    exit 0
  fi

  "${nvcc_bin}" -O3 -arch=sm_87 -Xcompiler "-fopenmp,-march=native" -o "$out" "$src"
  exit 0
fi

neon_flags=()
if grep -q "arm_neon\\.h" "$src" 2>/dev/null; then
  arch="$(uname -m)"
  if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
    echo "SKIP: NEON source ($src) on non-ARM host ($arch)"
    exit 0
  fi
  neon_flags+=("-march=armv8-a+simd")
else
  neon_flags+=("-march=native")
fi

"${CXX:-g++}" -O3 -std=c++17 -fopenmp "${neon_flags[@]}" -o "$out" "$src"
