#!/bin/bash
export PATH=/nfs/home/proj2_env/cmake/bin:/usr/local/cuda-11.4/bin:$PATH

set -e

cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/build_and_run.sh <log2_input_size>"
    exit 1
fi

bash scripts/build.sh
bash scripts/run.sh "$1"
