#!/bin/bash

set -e

cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/run.sh <log2_input_size>"
    exit 1
fi

if [ ! -x ./bin/prob2 ]; then
    echo "Binary not found. Build first with sbatch build_by_sbatch.sh or sbatch build_and_run_by_sbatch.sh <log2_input_size>."
    exit 1
fi

if [ -n "$SLURM_JOB_ID" ]; then
    srun --ntasks=1 ./bin/prob2 "$1"
else
    ./bin/prob2 "$1"
fi
