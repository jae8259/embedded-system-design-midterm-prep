#!/bin/bash

#SBATCH -J job
#SBATCH -o logs/out.job.%j
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

bash scripts/build_and_run.sh "$1"
