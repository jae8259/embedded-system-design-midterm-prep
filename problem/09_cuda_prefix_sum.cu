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
