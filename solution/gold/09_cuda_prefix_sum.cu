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

void prefix_sum(const float* d_in, float* d_out, float* d_block_sums, int num_blocks, int n) {
    int elems_per_block = BLOCK * 2;
    size_t smem = elems_per_block * sizeof(float);

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

    int elems_per_block = BLOCK * 2;
    int num_blocks = (N + elems_per_block - 1) / elems_per_block;
    float* d_block_sums_buf;
    cudaMalloc(&d_block_sums_buf, num_blocks * sizeof(float));

    prefix_sum(d_in, d_out, d_block_sums_buf, num_blocks, N);
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
    for (int i = 0; i < RUNS; i++) prefix_sum(d_in, d_out, d_block_sums_buf, num_blocks, N);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_gpu; cudaEventElapsedTime(&ms_gpu, ev_s, ev_e); ms_gpu /= RUNS;

    printf("TEST02:\nAVG %.2fms (cpu serial) vs AVG %.2fms (cuda scan)\n", ms_cpu, ms_gpu);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_block_sums_buf);
    delete[] h_in; delete[] h_ref; delete[] h_out;
    return 0;
}
