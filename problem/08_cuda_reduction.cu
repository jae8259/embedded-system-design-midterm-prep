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
