# Midterm Problem Set

Target board: Jetson Orin Nano (Ampere SM 8.2, 1024 CUDA cores)
Execution: `sbatch scripts/test.sh [--<problem-id>]`
Naming: `<ID>_<Description>.<cpp|cu>`

---

## Section C — Isolated Skills

Each problem focuses on exactly one optimization technique.
The skeleton contains a working serial baseline and a single `// TODO` block.

---

### P01 — OpenMP Parallel Reduction

**File:** `01_openmp_reduce.cpp`
**Difficulty:** Easy
**Dimension:** N = 2²⁴ (16 M) floats

Compute the sum of a large float array using OpenMP.
The serial version is provided. Parallelize the reduction with `#pragma omp parallel for reduction`.

**What to implement:** the `reduce_omp(float* data, int n)` function.
**Key concept:** OpenMP reduction clause, thread-safe accumulation.

**Tests:**
```
TEST01: Correctness — result matches serial sum within 1e-3
TEST02: AVG Xms (serial) vs AVG Yms (omp, 10 runs)
```

---

### P02 — NEON Vectorized Dot Product

**File:** `02_neon_dot.cpp`
**Difficulty:** Medium
**Dimension:** N = 2²⁴ (16 M) floats

Compute the dot product of two float arrays using ARM NEON SIMD intrinsics.
The serial version is provided. Replace the scalar loop with `float32x4_t` loads, `vmulq_f32`, `vaddq_f32`, and a horizontal reduction using `vaddvq_f32`.

**What to implement:** the `dot_neon(float* a, float* b, int n)` function.
**Key concept:** NEON 128-bit vector registers, 4-wide SIMD, horizontal reduce.

**Tests:**
```
TEST01: Correctness — result matches serial dot product within 1e-3
TEST02: AVG Xms (serial) vs AVG Yms (neon, 10 runs)
```

---

### P03 — CUDA Shared Memory Matrix Transpose

**File:** `03_cuda_transpose.cu`
**Difficulty:** Medium
**Dimension:** 1024 × 1024 floats

Transpose a square matrix on the GPU.
The naive CUDA transpose is provided (uncoalesced writes). Rewrite it using shared memory tiles to ensure coalesced global memory access in both reads and writes. Use a `(TILE+1)` padding column to avoid shared memory bank conflicts.

**What to implement:** `transpose_smem(float* in, float* out, int N)` kernel.
**Key concept:** shared memory tiling, coalesced access, bank conflict avoidance.

**Tests:**
```
TEST01: Correctness — output matches CPU reference transpose element-wise
TEST02: AVG Xms (naive kernel) vs AVG Yms (smem kernel, 10 runs)
```

---

### P04 — CUDA Warp Reduction with Shuffle

**File:** `04_cuda_warp_reduce.cu`
**Difficulty:** Medium
**Dimension:** N = 2²⁴ (16 M) floats

Sum a float array using only warp-level shuffle intrinsics — no shared memory.
A single-block naive kernel is provided. Implement a grid-stride kernel where each warp reduces its lane values with `__shfl_down_sync`, then atomically accumulates the warp sums into a global output.

**What to implement:** `reduce_warp(float* in, float* out, int n)` kernel.
**Key concept:** `__shfl_down_sync`, warp lanes, `atomicAdd` for cross-warp accumulation.

**Tests:**
```
TEST01: Correctness — result matches CPU sum within 1e-3
TEST02: AVG Xms (naive kernel) vs AVG Yms (warp-shuffle kernel, 10 runs)
```

---

### P05 — CUDA Histogram with Privatization

**File:** `05_cuda_histogram.cu`
**Difficulty:** Hard
**Dimension:** N = 2²⁴ values (uint8), 256 bins

Build a histogram of a byte array on the GPU.
A global-memory `atomicAdd` baseline is provided. Implement the privatized version: each block accumulates into a shared-memory local histogram, then merges into the global histogram at the end with `atomicAdd`.

**What to implement:** `histogram_privatized(uint8_t* in, int* hist, int n)` kernel.
**Key concept:** shared memory privatization, reducing atomic contention.

**Tests:**
```
TEST01: Correctness — bin counts match CPU reference histogram exactly
TEST02: AVG Xms (global atomic) vs AVG Yms (privatized, 10 runs)
```

---

## Section A — Progressive Complexity

Each problem combines multiple techniques. The skeleton provides the naive implementation and the function signature for the optimized version.

---

### P06 — Cache-Tiled CPU Matrix Transpose

**File:** `06_tiled_transpose.cpp`
**Difficulty:** Easy
**Dimension:** 4096 × 4096 floats

Transpose a large matrix on the CPU with a cache-friendly tiled loop.
The naive row-by-row transpose is provided (poor cache behavior on writes). Implement a tiled version that processes `TILE × TILE` blocks to improve L1/L2 reuse.

**What to implement:** `transpose_tiled(float* in, float* out, int N, int tile)` function.
**Key concept:** loop tiling, cache line reuse, row-major access patterns.

**Tests:**
```
TEST01: Correctness — output matches naive transpose element-wise
TEST02: AVG Xms (naive) vs AVG Yms (tiled, 10 runs)
```

---

### P07 — OpenMP Tiled Matrix Multiply

**File:** `07_omp_matmul.cpp`
**Difficulty:** Medium
**Dimension:** M = K = N = 512

Multiply two square matrices on the CPU using both loop tiling and OpenMP parallelism.
The naive triple-loop is provided. Implement a tiled version with `#pragma omp parallel for` on the outer tile loop.

**What to implement:** `matmul_omp(float* A, float* B, float* C, int M, int K, int N)` function.
**Key concept:** loop tiling for cache locality + OpenMP for thread-level parallelism.

**Tests:**
```
TEST01: Correctness — output matches naive matmul within 1e-3 per element
TEST02: AVG Xms (naive) vs AVG Yms (omp+tiled, 10 runs)
```

---

### P08 — Full CUDA Reduction Pipeline

**File:** `08_cuda_reduction.cu`
**Difficulty:** Hard
**Dimension:** N = 2²⁶ (64 M) floats

Reduce a very large float array to a scalar sum on the GPU using a full optimized pipeline.
A naive single-pass kernel is provided. Implement a two-pass reduction: first pass uses a grid-stride loop with shared memory block reduction and warp shuffle; second pass reduces the per-block partial sums. Also apply float4 vectorized loads.

**What to implement:** `reduce(float* in, float* out, int n)` host function.
**Key concepts:** grid-stride loop, shared memory, warp shuffle, float4 vectorization, two-pass reduction.

**Tests:**
```
TEST01: Correctness — result matches CPU sum within 1e-2
TEST02: AVG Xms (naive kernel) vs AVG Yms (optimized, 10 runs)
```

---

### P09 — CUDA Exclusive Prefix Sum (Blelloch Scan)

**File:** `09_cuda_prefix_sum.cu`
**Difficulty:** Hard
**Dimension:** N = 2²⁰ (1 M) floats

Compute the exclusive prefix sum (scan) of a float array on the GPU.
A serial CPU scan is provided for reference. Implement the Blelloch two-pass scan: up-sweep (reduce) builds a binary tree of partial sums in shared memory; down-sweep distributes them back. Handle arrays larger than one block with a multi-block approach (scan of block totals, then add back).

**What to implement:** `prefix_sum(float* in, float* out, int n)` host function.
**Key concept:** Blelloch up-sweep / down-sweep, shared memory, multi-block scan.

**Tests:**
```
TEST01: Correctness — output matches CPU serial scan element-wise within 1e-3
TEST02: AVG Xms (CPU serial) vs AVG Yms (CUDA scan, 10 runs)
```

---

### P10 — Hybrid CPU+GPU Matrix Multiply

**File:** `10_hybrid_matmul.cu`
**Difficulty:** Hard
**Dimension:** M = K = N = 1024

Implement matrix multiply twice — once on CPU with OpenMP+tiling, once on GPU with shared memory tiling — then benchmark both on the same input.

Two stubs are provided:
- `matmul_cpu(float* A, float* B, float* C, int N)` — use OpenMP + tiling (reuse P07 logic)
- `matmul_gpu(float* A, float* B, float* C, int N)` — CUDA tiled kernel with shared memory (BLOCK=32, KTILE=32)

**What to implement:** both functions above.
**Key concepts:** end-to-end CPU vs GPU pipeline, memory transfer cost awareness, choosing the right device.

**Tests:**
```
TEST01: Correctness — both CPU and GPU results match naive reference within 1e-2
TEST02: AVG Xms (cpu omp+tiled) vs AVG Yms (gpu tiled, 10 runs)
```

---

## Summary

| ID | File | Section | Topic | Difficulty |
|----|------|---------|-------|-----------|
| P01 | `01_openmp_reduce.cpp` | C | OpenMP reduction | Easy |
| P02 | `02_neon_dot.cpp` | C | NEON dot product | Medium |
| P03 | `03_cuda_transpose.cu` | C | CUDA shared mem transpose | Medium |
| P04 | `04_cuda_warp_reduce.cu` | C | Warp shuffle reduction | Medium |
| P05 | `05_cuda_histogram.cu` | C | CUDA histogram privatization | Hard |
| P06 | `06_tiled_transpose.cpp` | A | Cache-tiled CPU transpose | Easy |
| P07 | `07_omp_matmul.cpp` | A | OpenMP + tiled matmul | Medium |
| P08 | `08_cuda_reduction.cu` | A | Full CUDA reduction pipeline | Hard |
| P09 | `09_cuda_prefix_sum.cu` | A | Blelloch prefix scan | Hard |
| P10 | `10_hybrid_matmul.cu` | A | Hybrid CPU+GPU matmul | Hard |
