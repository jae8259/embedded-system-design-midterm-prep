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
    // Each thread adds up every gridDim elements; 256 x 256 list to reduce
    for (int i = idx; i < n; i += stride) sum += in[i];
    // Each warp reduces within them; (256 x 256) / 32 list to reduce
    sum = warp_reduce(sum);
    // Atomically add each warp result to the sum
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
