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
