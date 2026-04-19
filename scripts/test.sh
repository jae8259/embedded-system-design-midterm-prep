#!/bin/bash
#SBATCH -J midterm
#SBATCH -o logs/test.%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -e
PROBLEM=${1#--}   # strip leading -- if present

MINE_DIR="solution/mine"

compile_and_run() {
    local num=$1
    local cu_glob="$MINE_DIR/${num}_*.cu"
    local cpp_glob="$MINE_DIR/${num}_*.cpp"

    if ls $cu_glob 2>/dev/null | grep -q .; then
        local f=$(ls $cu_glob | head -1)
        nvcc -O3 -arch=sm_87 -Xcompiler "-fopenmp,-march=native" -o /tmp/p${num} "$f"
    elif ls $cpp_glob 2>/dev/null | grep -q .; then
        local f=$(ls $cpp_glob | head -1)
        g++ -O3 -std=c++17 -fopenmp -march=native -o /tmp/p${num} "$f"
    else
        echo "No solution file found for problem $num in $MINE_DIR"
        return 1
    fi

    echo "=== Problem $num ==="
    /tmp/p${num}
    echo ""
}

if [ -z "$PROBLEM" ]; then
    for num in 01 02 03 04 05 06 07 08 09 10; do
        compile_and_run "$num" || true
    done
else
    compile_and_run "$PROBLEM"
fi
