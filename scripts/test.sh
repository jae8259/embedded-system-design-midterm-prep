#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIDTERM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$MIDTERM_DIR/logs"

jobid="$(sbatch --parsable "$SCRIPT_DIR/test.sh" "$@")"
echo "Submitted batch job $jobid"
echo "Tailing: $MIDTERM_DIR/logs/slurm-midterm.${jobid}.out"

out="$MIDTERM_DIR/logs/slurm-midterm.${jobid}.out"
err="$MIDTERM_DIR/logs/slurm-midterm.${jobid}.err"

touch "$out" "$err"

tail -n +1 -F "$out" "$err" &
tail_pid=$!

cleanup() {
  kill "$tail_pid" 2>/dev/null || true
}
trap cleanup EXIT

while squeue -j "$jobid" -h >/dev/null 2>&1; do
  sleep 1
done

wait "$tail_pid" 2>/dev/null || true
