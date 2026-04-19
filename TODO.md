# Future Problems

Additional problems to generate after P01–P10 are complete.

---

## Section C — Isolated Skills (candidates)

- **NEON matrix-vector multiply** — `float32x4_t` tiled row × column, harder than dot product
- **CUDA coalesced vs uncoalesced benchmark** — write two kernels that differ only in access pattern, measure the gap
- **CUDA unified memory** — same computation with explicit `cudaMemcpy` vs `cudaMallocManaged`, compare overhead
- **OpenMP task parallelism** — tree reduction with `#pragma omp task` instead of reduction clause
- **CUDA stream pipeline** — overlap memory transfer and compute using two CUDA streams

## Section A — Progressive Complexity (candidates)

- **Parallel prefix sum (CPU)** — OpenMP-based scan using up/down sweep, compare to P09 GPU scan
- **CUDA 2D convolution** — naive vs shared memory tiled, small kernel (3×3, 5×5)
- **Sparse matrix-vector multiply (SpMV)** — CSR format on GPU with warp-per-row strategy
- **Bitonic sort** — CUDA parallel sort using warp shuffles, compare to CPU `std::sort`
- **CUDA matrix transpose — non-square** — extend P03/P06 to M×N with padding

## Section Hybrid (candidates)

- **Image blurring pipeline** — CPU loads image, GPU applies Gaussian blur, CPU saves result
- **CPU+GPU prefix sum comparison** — run P07-style CPU scan vs P09 GPU scan on same data, pick winner based on N

---

## Notes

- Skip: DNN, flash attention, sequence alignment (per Readme)
- Prefer problems solvable in ~30 min under exam pressure
- Each new problem needs: skeleton, gold solution, test harness, sbatch script
