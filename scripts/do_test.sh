#!/bin/bash
#SBATCH -J midterm
#SBATCH -o logs/slurm-%x.%j.out
#SBATCH -e logs/slurm-%x.%j.err
#SBATCH --cpus-per-task=6
#SBATCH --time=00:05:00

set -e
PROBLEM=${1#--} # strip leading -- if present

find_midterm_dir() {
  local base="${SLURM_SUBMIT_DIR:-$PWD}"
  local candidates=(
    "$base/midterm"
    "$base"
  )
  for d in "${candidates[@]}"; do
    if [[ -d "$d/solution" && -d "$d/problem" && -d "$d/scripts" ]]; then
      echo "$d"
      return 0
    fi
  done
  return 1
}

MIDTERM_DIR="$(find_midterm_dir || true)"
if [[ -z "$MIDTERM_DIR" ]]; then
  echo "ERROR: couldn't locate midterm dir from SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' PWD='$PWD'"
  echo "Expected either ./midterm/{scripts,problem,solution} or ./{scripts,problem,solution}."
  exit 2
fi

MINE_DIR="${MINE_DIR:-$MIDTERM_DIR/solution/mine}"
LOG_DIR="${LOG_DIR:-$MIDTERM_DIR/logs}"

mkdir -p "$LOG_DIR"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LOG_FILE="$LOG_DIR/test.${SLURM_JOB_ID}.log"
else
  LOG_FILE="$LOG_DIR/test.local.$(date +%s).log"
fi
exec > >(tee "$LOG_FILE") 2>&1

is_neon_source() {
  local f="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -q "arm_neon\\.h" "$f" 2>/dev/null
  else
    grep -q "arm_neon\\.h" "$f" 2>/dev/null
  fi
}

host_arch() {
  uname -m
}

compile_and_run() {
  local num=$1
  local cu_glob="$MINE_DIR/${num}_*.cu"
  local cpp_glob="$MINE_DIR/${num}_*.cpp"

  shopt -s nullglob
  local cu_files=($cu_glob)
  local cpp_files=($cpp_glob)
  shopt -u nullglob

  if ((${#cu_files[@]})); then
    local f="${cu_files[0]}"
    nvcc -O3 -arch=sm_87 -Xcompiler "-fopenmp,-march=native" -o /tmp/p${num} "$f"
  elif ((${#cpp_files[@]})); then
    local f="${cpp_files[0]}"
    if is_neon_source "$f"; then
      local arch
      arch="$(host_arch)"
      if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
        echo "=== Problem $num ==="
        echo "SKIP: $f requires ARM NEON; run this on the target (Jetson/aarch64)."
        echo ""
        return 0
      fi
      g++ -O3 -std=c++17 -fopenmp -march=armv8-a+simd -o /tmp/p${num} "$f"
    else
      g++ -O3 -std=c++17 -fopenmp -march=native -o /tmp/p${num} "$f"
    fi
  else
    echo "No solution file found for problem $num in $MINE_DIR"
    return 1
  fi

  echo "=== Problem $num ==="
  /tmp/p${num}
  echo ""
}

if [ -z "$PROBLEM" ]; then
  for num in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21; do
    compile_and_run "$num" || true
  done
else
  compile_and_run "$PROBLEM"
fi
