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
