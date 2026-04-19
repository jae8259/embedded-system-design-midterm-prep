#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N       = 1 << 24;
static const int NCHUNKS = 8;
static const int CHUNK   = N / NCHUNKS;
static const int BLOCK   = 256;
static const int GRID    = (CHUNK + BLOCK - 1) / BLOCK;

__global__ void scale_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.f;
}

static void run_sequential(float* h_in, float* h_out,
                           float* d_in, float* d_out, int n) {
    for (int c = 0; c < NCHUNKS; c++) {
        float* hi = h_in  + c * CHUNK;
        float* ho = h_out + c * CHUNK;
        cudaMemcpy(d_in, hi, CHUNK * sizeof(float), cudaMemcpyHostToDevice);
        scale_kernel<<<GRID, BLOCK>>>(d_in, d_out, CHUNK);
        cudaDeviceSynchronize();
        cudaMemcpy(ho, d_out, CHUNK * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

static void run_streamed(float* h_in, float* h_out,
                         float* d_in0, float* d_out0,
                         float* d_in1, float* d_out1, int n) {
    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);
    float* d_ins[2]  = {d_in0,  d_in1};
    float* d_outs[2] = {d_out0, d_out1};

    for (int c = 0; c < NCHUNKS; c++) {
        int si = c % 2;
        float* hi = h_in  + c * CHUNK;
        float* ho = h_out + c * CHUNK;
        cudaMemcpyAsync(d_ins[si],  hi, CHUNK * sizeof(float), cudaMemcpyHostToDevice, s[si]);
        scale_kernel<<<GRID, BLOCK, 0, s[si]>>>(d_ins[si], d_outs[si], CHUNK);
        cudaMemcpyAsync(ho, d_outs[si], CHUNK * sizeof(float), cudaMemcpyDeviceToHost, s[si]);
    }
    cudaStreamSynchronize(s[0]);
    cudaStreamSynchronize(s[1]);
    cudaStreamDestroy(s[0]);
    cudaStreamDestroy(s[1]);
}

int main() {
    float *h_in, *h_seq, *h_str;
    cudaMallocHost(&h_in,  N * sizeof(float));
    cudaMallocHost(&h_seq, N * sizeof(float));
    cudaMallocHost(&h_str, N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)i * 1e-6f;

    float *d_in0, *d_out0, *d_in1, *d_out1;
    cudaMalloc(&d_in0,  CHUNK * sizeof(float));
    cudaMalloc(&d_out0, CHUNK * sizeof(float));
    cudaMalloc(&d_in1,  CHUNK * sizeof(float));
    cudaMalloc(&d_out1, CHUNK * sizeof(float));

    run_sequential(h_in, h_seq, d_in0, d_out0, N);
    run_streamed  (h_in, h_str, d_in0, d_out0, d_in1, d_out1, N);

    printf("TEST01: CUDA stream pipeline correctness\n");
    bool ok = true;
    for (int i = 0; i < N && ok; i++)
        if (fabsf(h_str[i] - h_seq[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++)
        run_sequential(h_in, h_seq, d_in0, d_out0, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_seq = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++)
        run_streamed(h_in, h_str, d_in0, d_out0, d_in1, d_out1, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_str = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (sequential) vs AVG %.2fms (streamed)\n", ms_seq, ms_str);

    cudaFreeHost(h_in); cudaFreeHost(h_seq); cudaFreeHost(h_str);
    cudaFree(d_in0); cudaFree(d_out0); cudaFree(d_in1); cudaFree(d_out1);
    return 0;
}
