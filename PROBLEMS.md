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

---

## Section B — Additional Skills (from EXERCISE.md)

---

### P11 — std::thread Vector Addition

**File:** `11_thread_vecadd.cpp`
**Difficulty:** Easy
**Dimension:** N = 2²⁴ (16 M) floats

Add two float arrays element-wise on the CPU using `std::thread`.
The serial version is provided. Divide the array into NTHREADS equal chunks; launch one thread per chunk; join all threads.

**What to implement:** `vecadd_threads(float* a, float* b, float* c, int n)` function.
**Key concept:** `std::thread`, work partitioning, `join`.

**Tests:**
```
TEST01: Correctness — result matches serial addition element-wise
TEST02: AVG Xms (serial) vs AVG Yms (std::thread, 10 runs)
```

---

### P12 — NEON Vector Addition

**File:** `12_neon_vecadd.cpp`
**Difficulty:** Easy
**Dimension:** N = 2²⁴ (16 M) floats

Add two float arrays element-wise using ARM NEON SIMD.
The serial version is provided. Use `vld1q_f32`, `vaddq_f32`, `vst1q_f32` to process 4 floats per cycle; handle the tail scalar.

**What to implement:** `vecadd_neon(float* a, float* b, float* c, int n)` function.
**Key concept:** NEON store (`vst1q_f32`), 4-wide SIMD addition, scalar tail.

**Tests:**
```
TEST01: Correctness — result matches serial addition element-wise
TEST02: AVG Xms (serial) vs AVG Yms (neon, 10 runs)
```

---

### P13 — Thread-Safe Hashtable

**File:** `13_hashtable.cpp`
**Difficulty:** Hard
**Dimension:** 1M inserts, 8 threads, 1024 buckets (separate chaining)

Implement three locking strategies for a hash table with separate chaining.
The struct and helper methods are provided. Fill in the three insert functions.

**What to implement:** `insert_coarse`, `insert_fine`, `insert_strip` in `HashTable`.
**Key concept:** mutex granularity — global lock vs per-bucket lock vs lock striping.

**Tests:**
```
TEST01: Correctness — single-threaded insert produces expected size for all three methods
TEST02: AVG Xms (coarse) vs AVG Yms (fine) vs AVG Zms (strip, 10 runs concurrent)
```

---

### P14 — OpenMP Direct 2D Convolution

**File:** `14_omp_conv.cpp`
**Difficulty:** Medium
**Dimension:** 2048 × 2048 image, 3 × 3 kernel

Apply a 2D convolution (valid padding) to an image using OpenMP.
The serial four-nested-loop version is provided. Parallelize the outer pixel loops.

**What to implement:** `conv_omp(float* img, float* ker, float* out)` function.
**Key concept:** `#pragma omp parallel for collapse(2)`, embarrassingly parallel over output pixels.

**Tests:**
```
TEST01: Correctness — output matches serial convolution within 1e-4
TEST02: AVG Xms (serial) vs AVG Yms (omp, 10 runs)
```

---

### P15 — CUDA Unified Memory

**File:** `15_cuda_unified.cu`
**Difficulty:** Medium
**Dimension:** N = 2²⁴ (16 M) floats

Perform an element-wise scale (×2) using unified memory instead of explicit `cudaMemcpy`.
The explicit approach (cudaMalloc + cudaMemcpy + kernel + copy back) is provided.
Rewrite it with `cudaMallocManaged` and optional prefetch.

**What to implement:** `run_unified(float* h_in, float* h_out, int n)` function.
**Key concept:** `cudaMallocManaged`, `cudaMemPrefetchAsync`, programming model tradeoffs.

**Tests:**
```
TEST01: Correctness — unified result matches explicit approach element-wise
TEST02: AVG Xms (explicit) vs AVG Yms (unified, 10 runs, incl. transfers)
```

---

### P16 — CUDA Stream Pipeline

**File:** `16_cuda_stream.cu`
**Difficulty:** Hard
**Dimension:** N = 2²⁴ floats in 8 chunks

Overlap H2D transfer, compute, and D2H transfer across two CUDA streams.
The sequential single-stream approach (H2D → kernel → D2H per chunk, blocking) is provided.
Implement the two-stream ping-pong pipeline using `cudaMemcpyAsync`.

**What to implement:** `run_streamed(...)` function.
**Key concept:** `cudaStream_t`, `cudaMemcpyAsync`, `cudaMallocHost` (pinned), pipeline overlap.

**Tests:**
```
TEST01: Correctness — streamed output matches sequential output element-wise
TEST02: AVG Xms (sequential) vs AVG Yms (streamed, 10 runs)
```

---

---

## Section D — Combinations & Synthesis

---

### P17 — im2col + GEMM Convolution

**File:** `17_im2col_gemm.cpp`
**Difficulty:** Hard
**Dimension:** H=W=64, C=8, C_out=16, K=3

Transform a convolution into a matrix multiply via the im2col reshape.
The direct convolution and a `gemm` helper are provided. Implement `im2col` that unpacks input patches into a `[C*K*K][OH*OW]` column matrix; `conv_im2col` calls it then calls `gemm`.

**What to implement:** `im2col(float* input, float* col_buf)` function.
**Key concept:** im2col layout, patch extraction, convolution-as-GEMM.

**Tests:**
```
TEST01: Correctness — im2col+gemm matches direct conv within 1e-3
TEST02: AVG Xms (direct) vs AVG Yms (im2col+gemm, 10 runs)
```

---

### P18 — NEON Matrix Multiply

**File:** `18_neon_matmul.cpp`
**Difficulty:** Hard
**Dimension:** M=K=N=512

Multiply two square matrices using ARM NEON `vmlaq_f32` for 4-wide FMA.
A kij-ordered serial matmul is provided. Implement the NEON version with the same loop order: broadcast `A[i][k]` with `vdupq_n_f32`, accumulate into `C[i][j..j+3]` with `vmlaq_f32`.

**What to implement:** `matmul_neon(float* A, float* B, float* C, int m, int k, int n)` function.
**Key concept:** `vdupq_n_f32`, `vmlaq_f32`, `vld1q_f32`/`vst1q_f32`, kij SIMD.

**Tests:**
```
TEST01: Correctness — NEON result matches serial within 1e-2
TEST02: AVG Xms (serial) vs AVG Yms (neon, 10 runs)
```

---

### P19 — kij Loop-Order + Threaded Matmul

**File:** `19_kij_matmul.cpp`
**Difficulty:** Medium
**Dimension:** M=K=N=512, 8 threads

Two TODOs in one problem. First: reorder ijk → kij for cache-friendly B access. Second: parallelize with `std::thread` by partitioning the i-loop (row ranges) — unlike partitioning k which causes write races on C.

**What to implement:** `matmul_kij` (loop reorder) and `matmul_kij_thread` (std::thread, i-partition).
**Key concept:** loop order → cache behavior; safe thread partitioning without mutex.

**Tests:**
```
TEST01: kij SUCCESS + kij+thread SUCCESS (both match ijk within 1e-2)
TEST02: AVG Xms (ijk) vs AVG Yms (kij) vs AVG Zms (kij+thread, 10 runs)
```

---

### P20 — Depthwise Convolution

**File:** `20_dwconv.cpp`
**Difficulty:** Medium
**Dimension:** C=64, H=W=128, K=3

Parallelize a depthwise convolution with OpenMP. Unlike standard conv, each output channel c depends only on input channel c — channels are fully independent, making parallelization trivial.

**What to implement:** `dwconv_omp(float* input, float* kernels, float* output)` function.
**Key concept:** depthwise conv structure, channel independence, `#pragma omp parallel for`.

**Tests:**
```
TEST01: Correctness — omp result matches serial within 1e-4
TEST02: AVG Xms (serial) vs AVG Yms (omp, 10 runs)
```

---

### P21 — OpenMP Misc Constructs

**File:** `21_omp_misc.cpp`
**Difficulty:** Medium
**Dimension:** N=2²² (4M elements)

Showcase file for five OMP constructs. Two working demos are provided (`reduction`, `master`). Two TODOs to implement:

1. `find_max_parallel` — use `private(local_max)` + `#pragma omp critical` for global max
2. `histogram_atomic` — use `#pragma omp atomic` for race-free histogram updates

**What to implement:** `find_max_parallel` and `histogram_atomic`.
**Key concept:** `private`, `critical`, `atomic`, `master`, `omp_get_num_procs`.

**Tests:**
```
TEST01a: find_max correctness (private+critical)
TEST01b: histogram correctness (atomic)
TEST02: AVG Xms (serial hist) vs AVG Yms (atomic hist, 10 runs)
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
| P11 | `11_thread_vecadd.cpp` | B | std::thread vector addition | Easy |
| P12 | `12_neon_vecadd.cpp` | B | NEON vector addition | Easy |
| P13 | `13_hashtable.cpp` | B | Thread-safe hashtable | Hard |
| P14 | `14_omp_conv.cpp` | B | OpenMP direct convolution | Medium |
| P15 | `15_cuda_unified.cu` | B | CUDA unified memory | Medium |
| P16 | `16_cuda_stream.cu` | B | CUDA stream pipeline | Hard |
| P17 | `17_im2col_gemm.cpp` | D | im2col + GEMM convolution | Hard |
| P18 | `18_neon_matmul.cpp` | D | NEON matrix multiply | Hard |
| P19 | `19_kij_matmul.cpp` | D | kij loop order + std::thread | Medium |
| P20 | `20_dwconv.cpp` | D | Depthwise convolution | Medium |
| P21 | `21_omp_misc.cpp` | D | OpenMP misc constructs | Medium |
