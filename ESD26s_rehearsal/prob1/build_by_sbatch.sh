#!/bin/bash

#SBATCH -J build
#SBATCH -o logs/out.build.%j
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=6

bash scripts/build.sh