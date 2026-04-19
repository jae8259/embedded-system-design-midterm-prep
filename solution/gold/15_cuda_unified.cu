#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N       = 1 << 24;
static const int BLOCK   = 256;
static const int NBLOCKS = (N + BLOCK - 1) / BLOCK;

__global__ void scale_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.f;
}

static void run_explicit(const float* h_in, float* h_out, int n) {
    float* d_buf;
    cudaMalloc(&d_buf, n * sizeof(float));
    cudaMemcpy(d_buf, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    scale_kernel<<<NBLOCKS, BLOCK>>>(d_buf, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
}

static void run_unified(const float* h_in, float* h_out, int n) {
    float* um;
    cudaMallocManaged(&um, n * sizeof(float));
    memcpy(um, h_in, n * sizeof(float));
    cudaMemPrefetchAsync(um, n * sizeof(float), 0, 0); // prefetch to GPU 0
    scale_kernel<<<NBLOCKS, BLOCK>>>(um, n);
    cudaDeviceSynchronize();
    memcpy(h_out, um, n * sizeof(float));
    cudaFree(um);
}

int main() {
    float* h_in  = new float[N];
    float* h_ref = new float[N];
    float* h_um  = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)i * 1e-6f;

    run_explicit(h_in, h_ref, N);
    run_unified (h_in, h_um,  N);

    printf("TEST01: unified memory correctness\n");
    bool ok = true;
    for (int i = 0; i < N && ok; i++)
        if (fabsf(h_um[i] - h_ref[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) run_explicit(h_in, h_ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_exp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) run_unified(h_in, h_um, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_um = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (explicit) vs AVG %.2fms (unified)\n", ms_exp, ms_um);

    delete[] h_in; delete[] h_ref; delete[] h_um;
    return 0;
}
