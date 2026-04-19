# Midterm Problem Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 10 self-contained midterm practice problems (skeleton + gold solution + test harness) covering OpenMP, NEON SIMD, CUDA shared memory, warp intrinsics, atomics, cache tiling, and hybrid CPU+GPU.

**Architecture:** Each problem is a single self-contained source file containing (1) a reference/serial baseline already implemented, (2) an optimized `// TODO` function the student fills in, and (3) a `main()` with TEST01 correctness and TEST02 benchmark. The gold copy in `solution/gold/` is identical except the TODO is implemented. `scripts/test.sh` compiles and runs `solution/mine/` on the Jetson board via sbatch.

**Tech Stack:** C++17, OpenMP, ARM NEON (`arm_neon.h`), CUDA 12 (`nvcc -arch=sm_87`), Slurm/sbatch, Jetson Orin Nano (Ampere SM 8.7, 8-core ARM)

---

## File Map

| File | Role |
|------|------|
| `scripts/test.sh` | sbatch driver: compiles and runs a problem from `solution/mine/` |
| `problem/01_openmp_reduce.cpp` | Skeleton: OpenMP parallel reduction |
| `solution/gold/01_openmp_reduce.cpp` | Gold: same file, TODO implemented |
| `problem/02_neon_dot.cpp` | Skeleton: NEON dot product |
| `solution/gold/02_neon_dot.cpp` | Gold |
| `problem/03_cuda_transpose.cu` | Skeleton: CUDA shared memory transpose |
| `solution/gold/03_cuda_transpose.cu` | Gold |
| `problem/04_cuda_warp_reduce.cu` | Skeleton: warp shuffle reduction |
| `solution/gold/04_cuda_warp_reduce.cu` | Gold |
| `problem/05_cuda_histogram.cu` | Skeleton: histogram privatization |
| `solution/gold/05_cuda_histogram.cu` | Gold |
| `problem/06_tiled_transpose.cpp` | Skeleton: cache-tiled CPU transpose |
| `solution/gold/06_tiled_transpose.cpp` | Gold |
| `problem/07_omp_matmul.cpp` | Skeleton: OpenMP tiled matmul |
| `solution/gold/07_omp_matmul.cpp` | Gold |
| `problem/08_cuda_reduction.cu` | Skeleton: full CUDA reduction pipeline |
| `solution/gold/08_cuda_reduction.cu` | Gold |
| `problem/09_cuda_prefix_sum.cu` | Skeleton: Blelloch prefix scan |
| `solution/gold/09_cuda_prefix_sum.cu` | Gold |
| `problem/10_hybrid_matmul.cu` | Skeleton: CPU+GPU matmul |
| `solution/gold/10_hybrid_matmul.cu` | Gold |

**Verification note:** This project targets a Jetson Orin Nano board accessed via Slurm. Do NOT compile or run on the host Mac. Correctness is verified only by running `sbatch scripts/test.sh --NN` on the target board.

---

## Task 0: Write `scripts/test.sh`

**Files:**
- Create: `scripts/test.sh`

- [ ] **Step 1: Write the sbatch test driver**

```bash
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
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/test.sh
git add scripts/test.sh
git commit -m "feat: add sbatch test driver"
```

---

## Task 1: P01 — OpenMP Parallel Reduction

**Files:**
- Create: `problem/01_openmp_reduce.cpp`
- Create: `solution/gold/01_openmp_reduce.cpp`

- [ ] **Step 1: Write skeleton**

`problem/01_openmp_reduce.cpp`:
```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>

static const int N = 1 << 24; // 16M floats

static float reduce_serial(const float* data, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

// TODO: implement parallel reduction using OpenMP.
// Hint: #pragma omp parallel for reduction(+:sum)
static float reduce_omp(const float* data, int n) {
    float sum = 0.f;
    // TODO
    return sum;
}

int main() {
    float* data = new float[N];
    for (int i = 0; i < N; i++) data[i] = 1.f;

    float ref = reduce_serial(data, N);
    float res = reduce_omp(data, N);

    printf("TEST01: OpenMP reduce correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(res - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = reduce_serial(data, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = reduce_omp(data, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (omp)\n", ms_serial, ms_omp);

    delete[] data;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/01_openmp_reduce.cpp` — identical to skeleton except replace the TODO function body:

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>

static const int N = 1 << 24;

static float reduce_serial(const float* data, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

static float reduce_omp(const float* data, int n) {
    float sum = 0.f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

int main() {
    float* data = new float[N];
    for (int i = 0; i < N; i++) data[i] = 1.f;

    float ref = reduce_serial(data, N);
    float res = reduce_omp(data, N);

    printf("TEST01: OpenMP reduce correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(res - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = reduce_serial(data, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = reduce_omp(data, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (omp)\n", ms_serial, ms_omp);

    delete[] data;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/01_openmp_reduce.cpp solution/gold/01_openmp_reduce.cpp
git commit -m "feat: P01 OpenMP reduce skeleton and gold"
```

---

## Task 2: P02 — NEON Vectorized Dot Product

**Files:**
- Create: `problem/02_neon_dot.cpp`
- Create: `solution/gold/02_neon_dot.cpp`

- [ ] **Step 1: Write skeleton**

`problem/02_neon_dot.cpp`:
```cpp
#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N = 1 << 24; // 16M floats

static float dot_serial(const float* a, const float* b, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// TODO: implement dot product using ARM NEON SIMD intrinsics.
// Hints:
//   float32x4_t acc = vdupq_n_f32(0.f);
//   vld1q_f32(ptr)         — load 4 floats
//   vmlaq_f32(acc, va, vb) — acc += va * vb (fused multiply-accumulate)
//   vaddvq_f32(acc)        — horizontal sum of 4-wide vector
//   Handle tail (i < n) with scalar loop after SIMD loop.
static float dot_neon(const float* a, const float* b, int n) {
    // TODO
    return 0.f;
}

int main() {
    float* a = new float[N];
    float* b = new float[N];
    for (int i = 0; i < N; i++) { a[i] = 1.f; b[i] = 2.f; }

    float ref = dot_serial(a, b, N);
    float res = dot_neon(a, b, N);

    printf("TEST01: NEON dot product correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(res - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = dot_serial(a, b, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = dot_neon(a, b, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_neon = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (neon)\n", ms_serial, ms_neon);

    delete[] a; delete[] b;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/02_neon_dot.cpp`:
```cpp
#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N = 1 << 24;

static float dot_serial(const float* a, const float* b, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float dot_neon(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.f);
    int i = 0;
    for (; i <= n - 4; i += 4)
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

int main() {
    float* a = new float[N];
    float* b = new float[N];
    for (int i = 0; i < N; i++) { a[i] = 1.f; b[i] = 2.f; }

    float ref = dot_serial(a, b, N);
    float res = dot_neon(a, b, N);

    printf("TEST01: NEON dot product correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(res - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = dot_serial(a, b, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = dot_neon(a, b, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_neon = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (neon)\n", ms_serial, ms_neon);

    delete[] a; delete[] b;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/02_neon_dot.cpp solution/gold/02_neon_dot.cpp
git commit -m "feat: P02 NEON dot product skeleton and gold"
```

---

## Task 3: P03 — CUDA Shared Memory Matrix Transpose

**Files:**
- Create: `problem/03_cuda_transpose.cu`
- Create: `solution/gold/03_cuda_transpose.cu`

- [ ] **Step 1: Write skeleton**

`problem/03_cuda_transpose.cu`:
```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static const int N    = 1024;
static const int TILE = 32;

// Provided: naive transpose — uncoalesced global writes.
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) out[x * n + y] = in[y * n + x];
}

// TODO: implement using shared memory to make both reads and writes coalesced.
// Hints:
//   __shared__ float tile[TILE][TILE + 1];  // +1 avoids bank conflicts
//   Read phase: tile[threadIdx.y][threadIdx.x] = in[y * n + x];  (coalesced read)
//   __syncthreads();
//   Write phase: compute transposed output coordinates and write tile[threadIdx.x][threadIdx.y]
//   (swap threadIdx.x and threadIdx.y roles, and swap blockIdx.x/blockIdx.y for output coords)
__global__ void transpose_smem(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];
    // TODO
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float* h_in       = new float[N * N];
    float* h_naive    = new float[N * N];
    float* h_smem     = new float[N * N];
    for (int i = 0; i < N * N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    transpose_naive<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_naive, d_out, bytes, cudaMemcpyDeviceToHost);

    transpose_smem<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_smem, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("TEST01: CUDA smem transpose correctness\n");
    bool ok = true;
    for (int i = 0; i < N * N && ok; i++)
        if (fabsf(h_smem[i] - h_naive[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) transpose_naive<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_naive; cudaEventElapsedTime(&ms_naive, ev_s, ev_e); ms_naive /= RUNS;

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) transpose_smem<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_smem; cudaEventElapsedTime(&ms_smem, ev_s, ev_e); ms_smem /= RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (smem)\n", ms_naive, ms_smem);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_naive; delete[] h_smem;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/03_cuda_transpose.cu` — identical except replace `transpose_smem`:

```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static const int N    = 1024;
static const int TILE = 32;

__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) out[x * n + y] = in[y * n + x];
}

__global__ void transpose_smem(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    // transposed output block: blockIdx swapped
    int ox = blockIdx.y * TILE + threadIdx.x;
    int oy = blockIdx.x * TILE + threadIdx.y;
    if (ox < n && oy < n)
        out[oy * n + ox] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float* h_in    = new float[N * N];
    float* h_naive = new float[N * N];
    float* h_smem  = new float[N * N];
    for (int i = 0; i < N * N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    transpose_naive<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_naive, d_out, bytes, cudaMemcpyDeviceToHost);
    transpose_smem<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_smem, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("TEST01: CUDA smem transpose correctness\n");
    bool ok = true;
    for (int i = 0; i < N * N && ok; i++)
        if (fabsf(h_smem[i] - h_naive[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) transpose_naive<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_naive; cudaEventElapsedTime(&ms_naive, ev_s, ev_e); ms_naive /= RUNS;

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) transpose_smem<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_smem; cudaEventElapsedTime(&ms_smem, ev_s, ev_e); ms_smem /= RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (smem)\n", ms_naive, ms_smem);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_naive; delete[] h_smem;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/03_cuda_transpose.cu solution/gold/03_cuda_transpose.cu
git commit -m "feat: P03 CUDA smem transpose skeleton and gold"
```

---

## Task 4: P04 — CUDA Warp Shuffle Reduction

**Files:**
- Create: `problem/04_cuda_warp_reduce.cu`
- Create: `solution/gold/04_cuda_warp_reduce.cu`

- [ ] **Step 1: Write skeleton**

`problem/04_cuda_warp_reduce.cu`:
```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N       = 1 << 24; // 16M
static const int BLOCK   = 256;
static const int NBLOCKS = 256;

static float cpu_sum(const float* data, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += data[i];
    return s;
}

// TODO: use __shfl_down_sync to reduce val across all 32 lanes of a warp.
// Offsets: 16, 8, 4, 2, 1. mask = 0xffffffff.
// Return the sum held in lane 0 (other lanes have partial results — that's fine).
__device__ float warp_reduce(float val) {
    // TODO
    return val;
}

// TODO: implement grid-stride kernel.
// 1. Accumulate partial sum over all elements this thread owns (grid-stride loop).
// 2. Call warp_reduce(sum) to get warp-level sum.
// 3. If threadIdx.x % 32 == 0: atomicAdd(out, warp_sum).
__global__ void reduce_warp(const float* in, float* out, int n) {
    // TODO
}

int main() {
    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.f;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  (size_t)N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    reduce_warp<<<NBLOCKS, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    float h_out = 0.f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    float ref = cpu_sum(h_in, N);

    printf("TEST01: warp reduce correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(h_out - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = cpu_sum(h_in, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_out, 0, sizeof(float));
        reduce_warp<<<NBLOCKS, BLOCK>>>(d_in, d_out, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_gpu; cudaEventElapsedTime(&ms_gpu, ev_s, ev_e); ms_gpu /= RUNS;

    printf("TEST02:\nAVG %.2fms (cpu serial) vs AVG %.2fms (warp reduce)\n", ms_cpu, ms_gpu);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/04_cuda_warp_reduce.cu` — identical except replace `warp_reduce` and `reduce_warp`:

```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N       = 1 << 24;
static const int BLOCK   = 256;
static const int NBLOCKS = 256;

static float cpu_sum(const float* data, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += data[i];
    return s;
}

__device__ float warp_reduce(float val) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__global__ void reduce_warp(const float* in, float* out, int n) {
    float sum = 0.f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) sum += in[i];
    sum = warp_reduce(sum);
    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}

int main() {
    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.f;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  (size_t)N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    reduce_warp<<<NBLOCKS, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    float h_out = 0.f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    float ref = cpu_sum(h_in, N);

    printf("TEST01: warp reduce correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(h_out - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = cpu_sum(h_in, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_out, 0, sizeof(float));
        reduce_warp<<<NBLOCKS, BLOCK>>>(d_in, d_out, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_gpu; cudaEventElapsedTime(&ms_gpu, ev_s, ev_e); ms_gpu /= RUNS;

    printf("TEST02:\nAVG %.2fms (cpu serial) vs AVG %.2fms (warp reduce)\n", ms_cpu, ms_gpu);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/04_cuda_warp_reduce.cu solution/gold/04_cuda_warp_reduce.cu
git commit -m "feat: P04 CUDA warp shuffle reduce skeleton and gold"
```

---

## Task 5: P05 — CUDA Histogram Privatization

**Files:**
- Create: `problem/05_cuda_histogram.cu`
- Create: `solution/gold/05_cuda_histogram.cu`

- [ ] **Step 1: Write skeleton**

`problem/05_cuda_histogram.cu`:
```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const int N       = 1 << 24; // 16M
static const int BINS    = 256;
static const int BLOCK   = 256;
static const int NBLOCKS = 256;

// Provided: global-memory atomic histogram — high contention.
__global__ void histogram_global(const uint8_t* in, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        atomicAdd(&hist[in[i]], 1);
}

// TODO: implement privatized histogram to reduce atomic contention.
// Steps:
//   1. Declare __shared__ int local_hist[BINS]
//   2. Initialize: if (threadIdx.x < BINS) local_hist[threadIdx.x] = 0;  __syncthreads();
//   3. Grid-stride: atomicAdd to local_hist[in[i]]
//   4. __syncthreads();
//   5. Merge: if (threadIdx.x < BINS) atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
__global__ void histogram_privatized(const uint8_t* in, int* hist, int n) {
    // TODO
}

int main() {
    uint8_t* h_in = new uint8_t[N];
    for (int i = 0; i < N; i++) h_in[i] = (uint8_t)(i % BINS);

    uint8_t* d_in;
    int *d_global, *d_priv;
    cudaMalloc(&d_in,     N);
    cudaMalloc(&d_global, BINS * sizeof(int));
    cudaMalloc(&d_priv,   BINS * sizeof(int));
    cudaMemcpy(d_in, h_in, N, cudaMemcpyHostToDevice);

    cudaMemset(d_global, 0, BINS * sizeof(int));
    histogram_global<<<NBLOCKS, BLOCK>>>(d_in, d_global, N);

    cudaMemset(d_priv, 0, BINS * sizeof(int));
    histogram_privatized<<<NBLOCKS, BLOCK>>>(d_in, d_priv, N);
    cudaDeviceSynchronize();

    int h_global[BINS], h_priv[BINS];
    cudaMemcpy(h_global, d_global, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_priv,   d_priv,   BINS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("TEST01: histogram privatization correctness\n");
    bool ok = true;
    for (int i = 0; i < BINS && ok; i++)
        if (h_priv[i] != h_global[i]) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_global, 0, BINS * sizeof(int));
        histogram_global<<<NBLOCKS, BLOCK>>>(d_in, d_global, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_global; cudaEventElapsedTime(&ms_global, ev_s, ev_e); ms_global /= RUNS;

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_priv, 0, BINS * sizeof(int));
        histogram_privatized<<<NBLOCKS, BLOCK>>>(d_in, d_priv, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_priv; cudaEventElapsedTime(&ms_priv, ev_s, ev_e); ms_priv /= RUNS;

    printf("TEST02:\nAVG %.2fms (global atomic) vs AVG %.2fms (privatized)\n", ms_global, ms_priv);

    cudaFree(d_in); cudaFree(d_global); cudaFree(d_priv);
    delete[] h_in;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/05_cuda_histogram.cu` — identical except replace `histogram_privatized`:

```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const int N       = 1 << 24;
static const int BINS    = 256;
static const int BLOCK   = 256;
static const int NBLOCKS = 256;

__global__ void histogram_global(const uint8_t* in, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        atomicAdd(&hist[in[i]], 1);
}

__global__ void histogram_privatized(const uint8_t* in, int* hist, int n) {
    __shared__ int local_hist[BINS];
    if (threadIdx.x < BINS) local_hist[threadIdx.x] = 0;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        atomicAdd(&local_hist[in[i]], 1);
    __syncthreads();
    if (threadIdx.x < BINS)
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
}

int main() {
    uint8_t* h_in = new uint8_t[N];
    for (int i = 0; i < N; i++) h_in[i] = (uint8_t)(i % BINS);

    uint8_t* d_in;
    int *d_global, *d_priv;
    cudaMalloc(&d_in,     N);
    cudaMalloc(&d_global, BINS * sizeof(int));
    cudaMalloc(&d_priv,   BINS * sizeof(int));
    cudaMemcpy(d_in, h_in, N, cudaMemcpyHostToDevice);

    cudaMemset(d_global, 0, BINS * sizeof(int));
    histogram_global<<<NBLOCKS, BLOCK>>>(d_in, d_global, N);
    cudaMemset(d_priv, 0, BINS * sizeof(int));
    histogram_privatized<<<NBLOCKS, BLOCK>>>(d_in, d_priv, N);
    cudaDeviceSynchronize();

    int h_global[BINS], h_priv[BINS];
    cudaMemcpy(h_global, d_global, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_priv,   d_priv,   BINS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("TEST01: histogram privatization correctness\n");
    bool ok = true;
    for (int i = 0; i < BINS && ok; i++)
        if (h_priv[i] != h_global[i]) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_global, 0, BINS * sizeof(int));
        histogram_global<<<NBLOCKS, BLOCK>>>(d_in, d_global, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_global; cudaEventElapsedTime(&ms_global, ev_s, ev_e); ms_global /= RUNS;

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_priv, 0, BINS * sizeof(int));
        histogram_privatized<<<NBLOCKS, BLOCK>>>(d_in, d_priv, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_priv; cudaEventElapsedTime(&ms_priv, ev_s, ev_e); ms_priv /= RUNS;

    printf("TEST02:\nAVG %.2fms (global atomic) vs AVG %.2fms (privatized)\n", ms_global, ms_priv);

    cudaFree(d_in); cudaFree(d_global); cudaFree(d_priv);
    delete[] h_in;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/05_cuda_histogram.cu solution/gold/05_cuda_histogram.cu
git commit -m "feat: P05 CUDA histogram privatization skeleton and gold"
```

---

## Task 6: P06 — Cache-Tiled CPU Matrix Transpose

**Files:**
- Create: `problem/06_tiled_transpose.cpp`
- Create: `solution/gold/06_tiled_transpose.cpp`

- [ ] **Step 1: Write skeleton**

`problem/06_tiled_transpose.cpp`:
```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>

static const int N    = 4096;
static const int TILE = 64;

static void transpose_naive(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[j * n + i] = in[i * n + j];
}

// TODO: implement cache-tiled transpose to improve L1/L2 reuse.
// Hint: outer loops step by TILE (i += tile, j += tile),
//       inner loops ii in [i, min(i+tile, n)), jj in [j, min(j+tile, n))
//       out[jj*n+ii] = in[ii*n+jj]
static void transpose_tiled(const float* in, float* out, int n, int tile) {
    // TODO
}

int main() {
    float* in  = new float[(size_t)N * N];
    float* ref = new float[(size_t)N * N];
    float* out = new float[(size_t)N * N];
    for (int i = 0; i < N * N; i++) in[i] = (float)i;

    transpose_naive(in, ref, N);
    transpose_tiled(in, out, N, TILE);

    printf("TEST01: cache-tiled transpose correctness\n");
    bool ok = true;
    for (int i = 0; i < N * N && ok; i++)
        if (fabsf(out[i] - ref[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) transpose_naive(in, ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_naive = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) transpose_tiled(in, out, N, TILE);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_tiled = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (tiled)\n", ms_naive, ms_tiled);

    delete[] in; delete[] ref; delete[] out;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/06_tiled_transpose.cpp` — identical except replace `transpose_tiled`:

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>

static const int N    = 4096;
static const int TILE = 64;

static void transpose_naive(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[j * n + i] = in[i * n + j];
}

static void transpose_tiled(const float* in, float* out, int n, int tile) {
    for (int i = 0; i < n; i += tile)
        for (int j = 0; j < n; j += tile)
            for (int ii = i; ii < std::min(i + tile, n); ii++)
                for (int jj = j; jj < std::min(j + tile, n); jj++)
                    out[jj * n + ii] = in[ii * n + jj];
}

int main() {
    float* in  = new float[(size_t)N * N];
    float* ref = new float[(size_t)N * N];
    float* out = new float[(size_t)N * N];
    for (int i = 0; i < N * N; i++) in[i] = (float)i;

    transpose_naive(in, ref, N);
    transpose_tiled(in, out, N, TILE);

    printf("TEST01: cache-tiled transpose correctness\n");
    bool ok = true;
    for (int i = 0; i < N * N && ok; i++)
        if (fabsf(out[i] - ref[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) transpose_naive(in, ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_naive = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) transpose_tiled(in, out, N, TILE);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_tiled = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (tiled)\n", ms_naive, ms_tiled);

    delete[] in; delete[] ref; delete[] out;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/06_tiled_transpose.cpp solution/gold/06_tiled_transpose.cpp
git commit -m "feat: P06 cache-tiled CPU transpose skeleton and gold"
```

---

## Task 7: P07 — OpenMP Tiled Matrix Multiply

**Files:**
- Create: `problem/07_omp_matmul.cpp`
- Create: `solution/gold/07_omp_matmul.cpp`

- [ ] **Step 1: Write skeleton**

`problem/07_omp_matmul.cpp`:
```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

static const int M    = 512;
static const int K    = 512;
static const int NDIM = 512;
static const int TILE = 32;

// Row-major: A[M×K], B[K×N], C[M×N]
static void matmul_naive(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int l = 0; l < k; l++) sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = sum;
        }
}

// TODO: implement tiled matrix multiply with OpenMP parallelism.
// Hints:
//   memset(C, 0, m*n*sizeof(float));
//   #pragma omp parallel for collapse(2) schedule(static)
//   Outer tile loops: for i in [0,m) step TILE, for j in [0,n) step TILE
//   Inner sequential: for l in [0,k) step TILE
//   Innermost 3 loops: ii, jj, ll with min(i+TILE,m), min(j+TILE,n), min(l+TILE,k)
//   C[ii*n+jj] += A[ii*k+ll] * B[ll*n+jj]
static void matmul_omp(const float* A, const float* B, float* C, int m, int k, int n) {
    // TODO
}

int main() {
    float* A   = new float[M * K];
    float* B   = new float[K * NDIM];
    float* ref = new float[M * NDIM];
    float* out = new float[M * NDIM];
    for (int i = 0; i < M * K; i++) A[i] = 1.f;
    for (int i = 0; i < K * NDIM; i++) B[i] = 1.f;

    matmul_naive(A, B, ref, M, K, NDIM);
    matmul_omp(A, B, out, M, K, NDIM);

    printf("TEST01: OpenMP tiled matmul correctness\n");
    bool ok = true;
    for (int i = 0; i < M * NDIM && ok; i++)
        if (fabsf(out[i] - ref[i]) > 1e-2f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_naive(A, B, ref, M, K, NDIM);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_naive = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_omp(A, B, out, M, K, NDIM);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (omp+tiled)\n", ms_naive, ms_omp);

    delete[] A; delete[] B; delete[] ref; delete[] out;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/07_omp_matmul.cpp` — identical except replace `matmul_omp`:

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

static const int M    = 512;
static const int K    = 512;
static const int NDIM = 512;
static const int TILE = 32;

static void matmul_naive(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int l = 0; l < k; l++) sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = sum;
        }
}

static void matmul_omp(const float* A, const float* B, float* C, int m, int k, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; i += TILE)
        for (int j = 0; j < n; j += TILE)
            for (int l = 0; l < k; l += TILE)
                for (int ii = i; ii < std::min(i + TILE, m); ii++)
                    for (int jj = j; jj < std::min(j + TILE, n); jj++) {
                        float sum = 0.f;
                        for (int ll = l; ll < std::min(l + TILE, k); ll++)
                            sum += A[ii * k + ll] * B[ll * n + jj];
                        C[ii * n + jj] += sum;
                    }
}

int main() {
    float* A   = new float[M * K];
    float* B   = new float[K * NDIM];
    float* ref = new float[M * NDIM];
    float* out = new float[M * NDIM];
    for (int i = 0; i < M * K; i++) A[i] = 1.f;
    for (int i = 0; i < K * NDIM; i++) B[i] = 1.f;

    matmul_naive(A, B, ref, M, K, NDIM);
    matmul_omp(A, B, out, M, K, NDIM);

    printf("TEST01: OpenMP tiled matmul correctness\n");
    bool ok = true;
    for (int i = 0; i < M * NDIM && ok; i++)
        if (fabsf(out[i] - ref[i]) > 1e-2f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_naive(A, B, ref, M, K, NDIM);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_naive = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_omp(A, B, out, M, K, NDIM);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (omp+tiled)\n", ms_naive, ms_omp);

    delete[] A; delete[] B; delete[] ref; delete[] out;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/07_omp_matmul.cpp solution/gold/07_omp_matmul.cpp
git commit -m "feat: P07 OpenMP tiled matmul skeleton and gold"
```

---

## Task 8: P08 — Full CUDA Reduction Pipeline

**Files:**
- Create: `problem/08_cuda_reduction.cu`
- Create: `solution/gold/08_cuda_reduction.cu`

- [ ] **Step 1: Write skeleton**

`problem/08_cuda_reduction.cu`:
```cu
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N       = 1 << 26; // 64M floats
static const int BLOCK   = 512;
static const int NBLOCKS = 256;

static float cpu_sum(const float* data, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += data[i];
    return (float)s;
}

// Provided: naive shared-memory multi-block reduction (no float4, no warp shuffle).
__global__ void reduce_naive(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    s[tid] = (idx < n) ? in[idx] : 0.f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// TODO: implement warp-level reduction with __shfl_down_sync.
// Offsets 16, 8, 4, 2, 1; mask = 0xffffffff.
__device__ float warp_reduce_sum(float v) {
    // TODO
    return v;
}

// TODO: implement optimized reduction kernel.
// 1. Vectorized load: cast in to (const float4*), grid-stride over n/4 chunks, sum x+y+z+w.
// 2. Scalar tail: pick up remaining (n % 4) elements.
// 3. Store sum in shared mem s[tid], __syncthreads().
// 4. Shared-mem reduce down to 32 threads (stride > 32).
// 5. If tid < 32: load from s[tid], add s[tid+32] if BLOCK>=64, then warp_reduce_sum.
// 6. tid==0 writes to out[blockIdx.x].
__global__ void reduce_optimized(const float* in, float* out, int n) {
    extern __shared__ float s[];
    // TODO
}

// TODO: two-pass host function.
// Pass 1: reduce_optimized<<<NBLOCKS, BLOCK, BLOCK*sizeof(float)>>>(d_in, d_tmp, n)
// Pass 2: reduce_optimized<<<1,       BLOCK, BLOCK*sizeof(float)>>>(d_tmp, d_out, NBLOCKS)
void reduce(const float* d_in, float* d_out, float* d_tmp, int n) {
    // TODO
}

int main() {
    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.f;

    float *d_in, *d_out, *d_tmp;
    cudaMalloc(&d_in,  (size_t)N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMalloc(&d_tmp, NBLOCKS * sizeof(float));
    cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_out, 0, sizeof(float));
    reduce(d_in, d_out, d_tmp, N);
    cudaDeviceSynchronize();

    float h_out = 0.f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    float ref = cpu_sum(h_in, N);

    printf("TEST01: full CUDA reduction correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(h_out - ref) / ref < 1e-2f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    size_t smem = BLOCK * sizeof(float);
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        reduce_naive<<<NBLOCKS, BLOCK, smem>>>(d_in, d_tmp, N);
        reduce_naive<<<1, BLOCK, smem>>>(d_tmp, d_out, NBLOCKS);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_naive; cudaEventElapsedTime(&ms_naive, ev_s, ev_e); ms_naive /= RUNS;

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_out, 0, sizeof(float));
        reduce(d_in, d_out, d_tmp, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_opt; cudaEventElapsedTime(&ms_opt, ev_s, ev_e); ms_opt /= RUNS;

    printf("TEST02:\nAVG %.2fms (naive two-pass) vs AVG %.2fms (optimized)\n", ms_naive, ms_opt);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
    delete[] h_in;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/08_cuda_reduction.cu` — identical except replace `warp_reduce_sum`, `reduce_optimized`, and `reduce`:

```cu
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N       = 1 << 26;
static const int BLOCK   = 512;
static const int NBLOCKS = 256;

static float cpu_sum(const float* data, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += data[i];
    return (float)s;
}

__global__ void reduce_naive(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    s[tid] = (idx < n) ? in[idx] : 0.f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

__device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset);
    return v;
}

__global__ void reduce_optimized(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int grid_stride = blockDim.x * gridDim.x;
    float sum = 0.f;

    int vec_n = n / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (int i = global_idx; i < vec_n; i += grid_stride) {
        float4 v = in4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    for (int i = vec_n * 4 + global_idx; i < n; i += grid_stride)
        sum += in[i];

    s[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        sum = s[tid];
        if (BLOCK >= 64) sum += s[tid + 32];
        sum = warp_reduce_sum(sum);
        if (tid == 0) out[blockIdx.x] = sum;
    }
}

void reduce(const float* d_in, float* d_out, float* d_tmp, int n) {
    size_t smem = BLOCK * sizeof(float);
    reduce_optimized<<<NBLOCKS, BLOCK, smem>>>(d_in, d_tmp, n);
    reduce_optimized<<<1,       BLOCK, smem>>>(d_tmp, d_out, NBLOCKS);
}

int main() {
    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.f;

    float *d_in, *d_out, *d_tmp;
    cudaMalloc(&d_in,  (size_t)N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMalloc(&d_tmp, NBLOCKS * sizeof(float));
    cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_out, 0, sizeof(float));
    reduce(d_in, d_out, d_tmp, N);
    cudaDeviceSynchronize();

    float h_out = 0.f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    float ref = cpu_sum(h_in, N);

    printf("TEST01: full CUDA reduction correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(h_out - ref) / ref < 1e-2f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    size_t smem = BLOCK * sizeof(float);
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        reduce_naive<<<NBLOCKS, BLOCK, smem>>>(d_in, d_tmp, N);
        reduce_naive<<<1, BLOCK, smem>>>(d_tmp, d_out, NBLOCKS);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_naive; cudaEventElapsedTime(&ms_naive, ev_s, ev_e); ms_naive /= RUNS;

    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_out, 0, sizeof(float));
        reduce(d_in, d_out, d_tmp, N);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_opt; cudaEventElapsedTime(&ms_opt, ev_s, ev_e); ms_opt /= RUNS;

    printf("TEST02:\nAVG %.2fms (naive two-pass) vs AVG %.2fms (optimized)\n", ms_naive, ms_opt);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
    delete[] h_in;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/08_cuda_reduction.cu solution/gold/08_cuda_reduction.cu
git commit -m "feat: P08 full CUDA reduction pipeline skeleton and gold"
```

---

## Task 9: P09 — CUDA Exclusive Prefix Sum (Blelloch)

**Files:**
- Create: `problem/09_cuda_prefix_sum.cu`
- Create: `solution/gold/09_cuda_prefix_sum.cu`

- [ ] **Step 1: Write skeleton**

`problem/09_cuda_prefix_sum.cu`:
```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// N must be divisible by (BLOCK*2)
static const int N     = 1 << 20; // 1M floats
static const int BLOCK = 512;     // 2*BLOCK = 1024 elements per block

static void scan_cpu(const float* in, float* out, int n) {
    out[0] = 0.f;
    for (int i = 1; i < n; i++) out[i] = out[i - 1] + in[i - 1];
}

// TODO: Blelloch scan for one block segment (handles 2*BLOCK elements).
// Each block writes its total (before zeroing) to block_sums[blockIdx.x].
// Steps:
//   1. base = blockIdx.x * blockDim.x * 2
//   2. Load s[2*tid] and s[2*tid+1] from in[base+...], zero-pad if out of range.
//   3. Up-sweep: for stride = 1, 2, 4, ..., BLOCK:
//        __syncthreads()
//        idx = (tid+1)*stride*2 - 1
//        if idx < 2*BLOCK: s[idx] += s[idx - stride]
//   4. tid==0: block_sums[blockIdx.x] = s[2*BLOCK-1]; s[2*BLOCK-1] = 0.f
//   5. Down-sweep: for stride = BLOCK, BLOCK/2, ..., 1:
//        __syncthreads()
//        idx = (tid+1)*stride*2 - 1
//        if idx < 2*BLOCK: t=s[idx-stride]; s[idx-stride]=s[idx]; s[idx]+=t
//   6. __syncthreads(); write back to out[base+...]
__global__ void scan_blocks(const float* in, float* out, float* block_sums, int n) {
    extern __shared__ float s[];
    // TODO
}

// TODO: add block_sums[blockIdx.x] to every element owned by this block in out[].
// This block covers indices [blockIdx.x * blockDim.x, blockIdx.x * blockDim.x + blockDim.x).
// Use: int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < n) out[idx] += ...
__global__ void add_block_offsets(float* out, const float* offsets, int n) {
    // TODO
}

void prefix_sum(const float* d_in, float* d_out, int n) {
    int elems_per_block = BLOCK * 2;
    int num_blocks = (n + elems_per_block - 1) / elems_per_block;
    size_t smem = elems_per_block * sizeof(float);

    float* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    scan_blocks<<<num_blocks, BLOCK, smem>>>(d_in, d_out, d_block_sums, n);

    // Compute exclusive prefix offsets on CPU and upload
    float* h_sums = new float[num_blocks];
    cudaMemcpy(h_sums, d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float carry = 0.f;
    for (int i = 0; i < num_blocks; i++) {
        float total = h_sums[i];
        h_sums[i]   = carry;
        carry       += total;
    }
    cudaMemcpy(d_block_sums, h_sums, num_blocks * sizeof(float), cudaMemcpyHostToDevice);

    add_block_offsets<<<num_blocks, elems_per_block>>>(d_out, d_block_sums, n);

    cudaFree(d_block_sums);
    delete[] h_sums;
}

int main() {
    float* h_in  = new float[N];
    float* h_ref = new float[N];
    float* h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.f;
    scan_cpu(h_in, h_ref, N);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    prefix_sum(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("TEST01: CUDA prefix sum correctness\n");
    bool ok = true;
    for (int i = 0; i < N && ok; i++)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-2f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) scan_cpu(h_in, h_ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) prefix_sum(d_in, d_out, N);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_gpu; cudaEventElapsedTime(&ms_gpu, ev_s, ev_e); ms_gpu /= RUNS;

    printf("TEST02:\nAVG %.2fms (cpu serial) vs AVG %.2fms (cuda scan)\n", ms_cpu, ms_gpu);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_ref; delete[] h_out;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/09_cuda_prefix_sum.cu` — identical except replace `scan_blocks` and `add_block_offsets`:

```cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N     = 1 << 20;
static const int BLOCK = 512;

static void scan_cpu(const float* in, float* out, int n) {
    out[0] = 0.f;
    for (int i = 1; i < n; i++) out[i] = out[i - 1] + in[i - 1];
}

__global__ void scan_blocks(const float* in, float* out, float* block_sums, int n) {
    extern __shared__ float s[];
    int tid  = threadIdx.x;
    int len  = blockDim.x * 2;       // elements this block processes
    int base = blockIdx.x * len;

    s[2*tid]   = (base + 2*tid   < n) ? in[base + 2*tid]   : 0.f;
    s[2*tid+1] = (base + 2*tid+1 < n) ? in[base + 2*tid+1] : 0.f;

    // up-sweep
    for (int stride = 1; stride < len; stride <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < len) s[idx] += s[idx - stride];
    }
    if (tid == 0) { block_sums[blockIdx.x] = s[len - 1]; s[len - 1] = 0.f; }

    // down-sweep
    for (int stride = len >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < len) {
            float t = s[idx - stride];
            s[idx - stride] = s[idx];
            s[idx] += t;
        }
    }
    __syncthreads();
    if (base + 2*tid   < n) out[base + 2*tid]   = s[2*tid];
    if (base + 2*tid+1 < n) out[base + 2*tid+1] = s[2*tid+1];
}

__global__ void add_block_offsets(float* out, const float* offsets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] += offsets[blockIdx.x];
}

void prefix_sum(const float* d_in, float* d_out, int n) {
    int elems_per_block = BLOCK * 2;
    int num_blocks = (n + elems_per_block - 1) / elems_per_block;
    size_t smem = elems_per_block * sizeof(float);

    float* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    scan_blocks<<<num_blocks, BLOCK, smem>>>(d_in, d_out, d_block_sums, n);

    float* h_sums = new float[num_blocks];
    cudaMemcpy(h_sums, d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float carry = 0.f;
    for (int i = 0; i < num_blocks; i++) {
        float total = h_sums[i];
        h_sums[i]   = carry;
        carry       += total;
    }
    cudaMemcpy(d_block_sums, h_sums, num_blocks * sizeof(float), cudaMemcpyHostToDevice);

    add_block_offsets<<<num_blocks, elems_per_block>>>(d_out, d_block_sums, n);

    cudaFree(d_block_sums);
    delete[] h_sums;
}

int main() {
    float* h_in  = new float[N];
    float* h_ref = new float[N];
    float* h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.f;
    scan_cpu(h_in, h_ref, N);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    prefix_sum(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("TEST01: CUDA prefix sum correctness\n");
    bool ok = true;
    for (int i = 0; i < N && ok; i++)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-2f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) scan_cpu(h_in, h_ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    cudaEventRecord(ev_s);
    for (int i = 0; i < RUNS; i++) prefix_sum(d_in, d_out, N);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_gpu; cudaEventElapsedTime(&ms_gpu, ev_s, ev_e); ms_gpu /= RUNS;

    printf("TEST02:\nAVG %.2fms (cpu serial) vs AVG %.2fms (cuda scan)\n", ms_cpu, ms_gpu);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_ref; delete[] h_out;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/09_cuda_prefix_sum.cu solution/gold/09_cuda_prefix_sum.cu
git commit -m "feat: P09 CUDA Blelloch prefix sum skeleton and gold"
```

---

## Task 10: P10 — Hybrid CPU+GPU Matrix Multiply

**Files:**
- Create: `problem/10_hybrid_matmul.cu`
- Create: `solution/gold/10_hybrid_matmul.cu`

- [ ] **Step 1: Write skeleton**

`problem/10_hybrid_matmul.cu`:
```cu
#include <cuda_runtime.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

static const int N        = 1024;
static const int CPU_TILE = 32;
static const int GPU_BLK  = 32;  // GPU thread block side and K-tile size

// Provided: naive CPU matmul for correctness reference.
static void matmul_naive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int k = 0; k < n; k++) sum += A[i*n+k] * B[k*n+j];
            C[i*n+j] = sum;
        }
}

// TODO: implement CPU matmul with OpenMP + cache tiling.
// Hint: same structure as P07 matmul_omp, but n×n square.
//   memset C to 0
//   #pragma omp parallel for collapse(2) schedule(static)  on i,j tile loops
//   inner sequential k tile loop, then ii/jj/kk loops with min(x+CPU_TILE, n)
static void matmul_cpu(const float* A, const float* B, float* C, int n) {
    // TODO
}

// TODO: implement tiled CUDA matmul kernel.
// Each thread block computes a GPU_BLK × GPU_BLK output tile.
// Hints:
//   __shared__ float As[GPU_BLK][GPU_BLK], Bs[GPU_BLK][GPU_BLK];
//   Loop over K tiles: load A tile into As and B tile into Bs cooperatively.
//   As[threadIdx.y][threadIdx.x] = A[row * n + (t*GPU_BLK + threadIdx.x)]
//   Bs[threadIdx.y][threadIdx.x] = B[(t*GPU_BLK + threadIdx.y) * n + col]
//   __syncthreads(); accumulate dot product; __syncthreads()
//   Write C[row*n+col] = sum at the end.
__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[GPU_BLK][GPU_BLK];
    __shared__ float Bs[GPU_BLK][GPU_BLK];
    // TODO
}

static void matmul_gpu(const float* h_A, const float* h_B, float* h_C, int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    dim3 block(GPU_BLK, GPU_BLK);
    dim3 grid((n + GPU_BLK - 1) / GPU_BLK, (n + GPU_BLK - 1) / GPU_BLK);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    size_t sz  = (size_t)N * N * sizeof(float);
    float* A       = new float[N * N];
    float* B       = new float[N * N];
    float* ref     = new float[N * N];
    float* out_cpu = new float[N * N];
    float* out_gpu = new float[N * N];
    for (int i = 0; i < N * N; i++) { A[i] = 1.f; B[i] = 1.f; }

    matmul_naive(A, B, ref, N);
    matmul_cpu(A, B, out_cpu, N);
    matmul_gpu(A, B, out_gpu, N);

    printf("TEST01: hybrid matmul correctness\n");
    bool ok_cpu = true, ok_gpu = true;
    for (int i = 0; i < N * N; i++) {
        if (fabsf(out_cpu[i] - ref[i]) > 1e-2f) ok_cpu = false;
        if (fabsf(out_gpu[i] - ref[i]) > 1e-2f) ok_gpu = false;
    }
    printf("CPU: %s  GPU: %s\n",
           ok_cpu ? "SUCCESS" : "FAIL",
           ok_gpu ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_cpu(A, B, out_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_gpu(A, B, out_gpu, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (cpu omp+tiled) vs AVG %.2fms (gpu tiled)\n", ms_cpu, ms_gpu);

    delete[] A; delete[] B; delete[] ref; delete[] out_cpu; delete[] out_gpu;
    return 0;
}
```

- [ ] **Step 2: Write gold**

`solution/gold/10_hybrid_matmul.cu` — identical except replace `matmul_cpu` and `matmul_kernel`:

```cu
#include <cuda_runtime.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

static const int N        = 1024;
static const int CPU_TILE = 32;
static const int GPU_BLK  = 32;

static void matmul_naive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int k = 0; k < n; k++) sum += A[i*n+k] * B[k*n+j];
            C[i*n+j] = sum;
        }
}

static void matmul_cpu(const float* A, const float* B, float* C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(float));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += CPU_TILE)
        for (int j = 0; j < n; j += CPU_TILE)
            for (int k = 0; k < n; k += CPU_TILE)
                for (int ii = i; ii < std::min(i + CPU_TILE, n); ii++)
                    for (int jj = j; jj < std::min(j + CPU_TILE, n); jj++) {
                        float sum = 0.f;
                        for (int kk = k; kk < std::min(k + CPU_TILE, n); kk++)
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        C[ii * n + jj] += sum;
                    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[GPU_BLK][GPU_BLK];
    __shared__ float Bs[GPU_BLK][GPU_BLK];
    int row = blockIdx.y * GPU_BLK + threadIdx.y;
    int col = blockIdx.x * GPU_BLK + threadIdx.x;
    float sum = 0.f;
    int num_tiles = (n + GPU_BLK - 1) / GPU_BLK;
    for (int t = 0; t < num_tiles; t++) {
        int ak = t * GPU_BLK + threadIdx.x;
        int bk = t * GPU_BLK + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && ak < n) ? A[row * n + ak] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk  < n && col < n) ? B[bk  * n + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < GPU_BLK; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

static void matmul_gpu(const float* h_A, const float* h_B, float* h_C, int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    dim3 block(GPU_BLK, GPU_BLK);
    dim3 grid((n + GPU_BLK - 1) / GPU_BLK, (n + GPU_BLK - 1) / GPU_BLK);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    float* A       = new float[N * N];
    float* B       = new float[N * N];
    float* ref     = new float[N * N];
    float* out_cpu = new float[N * N];
    float* out_gpu = new float[N * N];
    for (int i = 0; i < N * N; i++) { A[i] = 1.f; B[i] = 1.f; }

    matmul_naive(A, B, ref, N);
    matmul_cpu(A, B, out_cpu, N);
    matmul_gpu(A, B, out_gpu, N);

    printf("TEST01: hybrid matmul correctness\n");
    bool ok_cpu = true, ok_gpu = true;
    for (int i = 0; i < N * N; i++) {
        if (fabsf(out_cpu[i] - ref[i]) > 1e-2f) ok_cpu = false;
        if (fabsf(out_gpu[i] - ref[i]) > 1e-2f) ok_gpu = false;
    }
    printf("CPU: %s  GPU: %s\n",
           ok_cpu ? "SUCCESS" : "FAIL",
           ok_gpu ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_cpu(A, B, out_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_gpu(A, B, out_gpu, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (cpu omp+tiled) vs AVG %.2fms (gpu tiled)\n", ms_cpu, ms_gpu);

    delete[] A; delete[] B; delete[] ref; delete[] out_cpu; delete[] out_gpu;
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
git add problem/10_hybrid_matmul.cu solution/gold/10_hybrid_matmul.cu
git commit -m "feat: P10 hybrid CPU+GPU matmul skeleton and gold"
```

---

## Spec Coverage Check

| Requirement | Covered by |
|------------|-----------|
| 5 isolated-skill problems (Section C) | Tasks 1–5 |
| 5 progressive problems (Section A) | Tasks 6–10 |
| Difficulty field | PROBLEMS.md |
| Dimensions specified per problem | Each file's constants |
| skeleton with // TODO | All problem/ files |
| Gold in solution/gold/ | All tasks |
| TEST01 correctness, TEST02 benchmark (10 runs) | main() in every file |
| Output format: TEST01/SUCCESS, TEST02/AVG Xms vs AVG Yms | main() in every file |
| sbatch scripts/test.sh --NN | Task 0 |
| Naming: ID_Description.ext | All files follow pattern |
