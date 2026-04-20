#!/bin/bash

#SBATCH -J exec
#SBATCH -o logs/out.exec.%j
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=6

bash scripts/run.sh