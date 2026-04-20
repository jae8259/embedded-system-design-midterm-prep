# Problem 2: Prefix Sum on GPU 

## 1. Overview
In this problem, your goal is to accelerate a 1D **inclusive prefix sum** using CUDA C++ on the Jetson Orin Nano.
You must **modify only** `src/prefix_sum.cu`.
You will be graded on both correctness and performance for **three test cases** (see 6. Grading for details).
Use **GPU (CUDA)**. The provided skeleton in `src/prefix_sum.cu` is correct but very slow; replace it with a faster CUDA implementation.


## 2. Problem Explanation

Given an input array `input` of length `N`, produce an output array `output` such that:

`output[i] = input[0] + input[1] + ... + input[i]`

This problem uses the **inclusive** prefix sum definition:
- `output[0] = input[0]`
- `output[1] = input[0] + input[1]`
- `output[2] = input[0] + input[1] + input[2]`

Example:

```text
input  = [3, 1, 4, 2]
output = [3, 4, 8, 10]
```

- We will test the functionality and speedup of **only the following three** input sizes: `2^18`,`2^19`, and `2^20`.
- The driver program (`src/main.cpp`) accepts `log2_input_size` in the range `[2, 28]` and verifies correctness against a CPU reference implementation.
- The driver allocates input/output on the host. Your implementation is responsible for any GPU-side memory management and kernel launches.

## 3. Submission Files

- When grading, we will use only your `src/prefix_sum.cu`.
- Other file modifications (e.g., modifying `src/main.cpp`) will not be reflected during grading.

## 4. Directory Description

The directory contains:
```
.
├── Makefile                         # CUDA build entry used by the helper scripts
├── build_by_sbatch.sh               # Slurm script for build only
├── build_and_run_by_sbatch.sh       # Slurm script for build and run in one job
├── run_by_sbatch.sh                 # Slurm script for run only
├── bin/                             # Output directory for built binaries (automatically generated)
├── logs/                          # Slurm output logs (automatically generated)
├── scripts/                         # Helper scripts used by the Slurm entrypoints
├── support/                         # Headers shared by the sources
└── src/
    ├── main.cpp                     # Generates test cases, checks correctness, measures execution time (do not modify)
    └── prefix_sum.cu                # Your implementation file; edit this file only
```

There is no input data directory here. All input arrays are generated at runtime by `main.cpp`.

## 5. How to Build and Run

**Use Slurm only.** 

The executable takes the input size in `log2` form.

Examples:
- `10` means the input length is `2^10`
- `20` means the input length is `2^20`

Build:

```bash
sbatch build_by_sbatch.sh
```

Run (after building):

```bash
sbatch run_by_sbatch.sh 20
```

Build and run in one job:

```bash
sbatch build_and_run_by_sbatch.sh 20
```

For each run, the program prints:
- input log2 size
- input size
- elapsed time in milliseconds
- `Correctness: PASS` or `Correctness: FAIL`



## 6. Grading

1. Test Cases
- Three randomly generated test cases with fixed input sizes (`2^18`, `2^19`, and `2^20`) will be used.
- **You only need to consider correctness and performance for these three input sizes.**

2. Correctness
- The output must pass the functionality test.
- Code that generates incorrect results will receive a score of 0.

3. **Performance**
- Your implementation has to be faster than the execution time specified below for each test case. 
- `2^18`: **300 ms**
- `2^19`: **275 ms**
- `2^20`: **250 ms**
