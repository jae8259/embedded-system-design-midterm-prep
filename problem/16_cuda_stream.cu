#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N       = 1 << 24; // 16M floats total
static const int NCHUNKS = 8;
static const int CHUNK   = N / NCHUNKS; // 2M floats per chunk
static const int BLOCK   = 256;
static const int GRID    = (CHUNK + BLOCK - 1) / BLOCK;

__global__ void scale_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.f;
}

// Provided: sequential — for each chunk, H2D → kernel → D2H (all blocking, one device buffer)
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

// TODO: implement two-stream pipeline to overlap H2D, compute, and D2H across chunks.
// Two device buffer pairs (d_in0/d_out0, d_in1/d_out1) and two streams ping-pong by chunk index.
// Steps:
//   1. cudaStreamCreate(&s[0]) and (&s[1])
//   2. For each chunk c: pick si = c % 2, select d_ins[si]/d_outs[si]
//      a. cudaMemcpyAsync(d_ins[si], h_in+c*CHUNK, ..., cudaMemcpyHostToDevice, s[si])
//      b. scale_kernel<<<GRID, BLOCK, 0, s[si]>>>(d_ins[si], d_outs[si], CHUNK)
//      c. cudaMemcpyAsync(h_out+c*CHUNK, d_outs[si], ..., cudaMemcpyDeviceToHost, s[si])
//   3. cudaStreamSynchronize(s[0]); cudaStreamSynchronize(s[1]);
//   4. cudaStreamDestroy both streams.
// Note: h_in and h_out must be pinned (cudaMallocHost) for async overlap to work.
static void run_streamed(float* h_in, float* h_out,
                         float* d_in0, float* d_out0,
                         float* d_in1, float* d_out1, int n) {
    // TODO
}

int main() {
    // Use pinned host memory so cudaMemcpyAsync can truly overlap with computation.
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
