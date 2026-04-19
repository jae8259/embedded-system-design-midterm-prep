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
