#!/bin/bash

#SBATCH -J build
#SBATCH -o logs/out.build.%j
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

bash scripts/build.sh
