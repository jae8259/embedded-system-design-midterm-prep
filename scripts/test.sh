#!/bin/bash
#SBATCH -J midterm
#SBATCH -o logs/slurm-%x.%j.out
#SBATCH -e logs/slurm-%x.%j.err
#SBATCH --cpus-per-task=6
#SBATCH --time=00:05:00

set -euo pipefail
PROBLEM="${1:-}"
PROBLEM="${PROBLEM#--}" # strip leading -- if present

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
  echo "ERROR: couldn't locate midterm dir from SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}' PWD='$PWD'"
  echo "Expected either ./midterm/{Makefile,scripts,problem,solution} or ./{Makefile,scripts,problem,solution}."
  exit 2
fi

LOG_DIR="$MIDTERM_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LOG_FILE="$LOG_DIR/test.${SLURM_JOB_ID}.log"
else
  LOG_FILE="$LOG_DIR/test.local.$(date +%s).log"
fi
exec > >(tee "$LOG_FILE") 2>&1

run_problem() {
  local num="$1"
  local bin="$MIDTERM_DIR/bin/p${num}"

  if [[ -f "$MIDTERM_DIR/Makefile" ]]; then
    if ! make -C "$MIDTERM_DIR" "p${num}"; then
      echo "=== Problem $num ==="
      echo "BUILD FAIL"
      echo ""
      return 1
    fi
  elif [[ -x "$MIDTERM_DIR/scripts/build_one.sh" ]]; then
    if ! "$MIDTERM_DIR/scripts/build_one.sh" "$num" "$bin"; then
      echo "=== Problem $num ==="
      echo "BUILD FAIL"
      echo ""
      return 1
    fi
  else
    echo "=== Problem $num ==="
    echo "BUILD FAIL: missing $MIDTERM_DIR/Makefile (and no $MIDTERM_DIR/scripts/build_one.sh fallback)"
    echo ""
    return 1
  fi

  if [[ ! -x "$bin" ]]; then
    echo "=== Problem $num ==="
    echo "SKIP: no runnable binary produced at $bin"
    echo ""
    return 0
  fi

  echo "=== Problem $num ==="
  "$bin"
  echo ""
}

if [[ -z "$PROBLEM" ]]; then
  for num in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21; do
    run_problem "$num" || true
  done
else
  run_problem "$PROBLEM"
fi
