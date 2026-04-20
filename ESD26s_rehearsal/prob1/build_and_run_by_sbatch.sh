#!/bin/bash

#SBATCH -J job
#SBATCH -o logs/out.job.%j
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=6

bash scripts/build_and_run.sh