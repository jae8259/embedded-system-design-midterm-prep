# Problem 1: Matrix Multiplication on CPU

## 1. Overview
In this problem, your goal is to optimize a CPU implementation of **matrix multiplication** using SIMD and multithreading.
You must **modify only** `src/matmul.cpp`.
You will be graded on both correctness and performance for **three test cases** (see 6. Grading for details).
Use **CPU only** with SIMD/multithreading. GPU usage is **not allowed**.


## 2. Problem Explanation

Given two input matrices `A` and `B`, your program must compute the matrix product `C = AB`.

- The driver program (`src/main.cpp`) loads input matrices from the `data/` directory, invokes your implementation, and verifies correctness against the provided reference output.
- Test cases include square and rectangular shapes (`1024x1024`, `2048x6114`, `791x1113`).

## 3. Submission Files

- When grading, we will use only your `src/matmul.cpp`.
- Other file modifications (e.g., modifying `src/main.cpp`) will not be reflected during grading.

## 4. Directory Description

The directory contains:
```
.
├── Makefile                         # Build entry used by the helper scripts
├── build_by_sbatch.sh               # Slurm script for build only
├── build_and_run_by_sbatch.sh       # Slurm script for build and run in one job
├── run_by_sbatch.sh                 # Slurm script for run only
├── bin/                             # Output directory for built binaries (automatically generated)
├── data/                            # Input/output matrix files used for testing
├── logs/                            # Slurm output logs (automatically generated)
├── scripts/                         # Helper scripts used by the Slurm entrypoints
├── support/                         # Headers shared by the sources
└── src/
    ├── main.cpp                     # Loads inputs, checks correctness, measures execution time (do not modify)
    └── matmul.cpp                   # Your implementation file; edit this file only
```

## 5. How to Build and Run

**Use Slurm only.**

In order to utilize multithreading, do **not** change the `--cpus-per-task` parameter inside `run_by_sbatch.sh`.

Build:

```bash
sbatch build_by_sbatch.sh
```

Run (after building):

```bash
sbatch run_by_sbatch.sh
```

Build and run in one job:

```bash
sbatch build_and_run_by_sbatch.sh
```

For each run, the program prints:
- input size
- elapsed time in milliseconds
- `Correctness: PASS` or `Correctness: FAIL`



## 6. Grading

1. Test Cases
- Three randomly generated test cases with fixed input sizes (`1024x1024`, `2048x6114`, `791x1113`) will be used.
- **You only need to consider correctness and performance for these three input sizes.**

2. Correctness
- The output must pass the functionality test.
- Code that generates incorrect results will receive a score of 0.

3. **Performance**
- Your implementation has to be faster than the execution time specified below for each test case.
- `791x1113`: **45 ms**
- `1024x1024`: **60 ms**
- `2048x6114`: **1450 ms**
