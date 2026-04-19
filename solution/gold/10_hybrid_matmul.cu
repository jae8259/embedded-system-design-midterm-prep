#include <cuda_runtime.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

static const int N        = 1024;
static const int CPU_TILE = 32;
static const int GPU_BLK  = 32;

static void matmul_naive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int k = 0; k < n; k++) sum += A[i*n+k] * B[k*n+j];
            C[i*n+j] = sum;
        }
}

static void matmul_cpu(const float* A, const float* B, float* C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(float));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += CPU_TILE)
        for (int j = 0; j < n; j += CPU_TILE)
            for (int k = 0; k < n; k += CPU_TILE)
                for (int ii = i; ii < std::min(i + CPU_TILE, n); ii++)
                    for (int jj = j; jj < std::min(j + CPU_TILE, n); jj++) {
                        float sum = 0.f;
                        for (int kk = k; kk < std::min(k + CPU_TILE, n); kk++)
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        C[ii * n + jj] += sum;
                    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[GPU_BLK][GPU_BLK];
    __shared__ float Bs[GPU_BLK][GPU_BLK];
    int row = blockIdx.y * GPU_BLK + threadIdx.y;
    int col = blockIdx.x * GPU_BLK + threadIdx.x;
    float sum = 0.f;
    int num_tiles = (n + GPU_BLK - 1) / GPU_BLK;
    for (int t = 0; t < num_tiles; t++) {
        int ak = t * GPU_BLK + threadIdx.x;
        int bk = t * GPU_BLK + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && ak < n) ? A[row * n + ak] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk  < n && col < n) ? B[bk  * n + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < GPU_BLK; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

static void matmul_gpu(const float* h_A, const float* h_B, float* h_C, int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    dim3 block(GPU_BLK, GPU_BLK);
    dim3 grid((n + GPU_BLK - 1) / GPU_BLK, (n + GPU_BLK - 1) / GPU_BLK);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    float* A       = new float[N * N];
    float* B       = new float[N * N];
    float* ref     = new float[N * N];
    float* out_cpu = new float[N * N];
    float* out_gpu = new float[N * N];
    for (int i = 0; i < N * N; i++) { A[i] = 1.f; B[i] = 1.f; }

    matmul_naive(A, B, ref, N);
    matmul_cpu(A, B, out_cpu, N);
    matmul_gpu(A, B, out_gpu, N);

    printf("TEST01: hybrid matmul correctness\n");
    bool ok_cpu = true, ok_gpu = true;
    for (int i = 0; i < N * N; i++) {
        if (fabsf(out_cpu[i] - ref[i]) > 1e-2f) ok_cpu = false;
        if (fabsf(out_gpu[i] - ref[i]) > 1e-2f) ok_gpu = false;
    }
    printf("CPU: %s  GPU: %s\n",
           ok_cpu ? "SUCCESS" : "FAIL",
           ok_gpu ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_cpu(A, B, out_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_gpu(A, B, out_gpu, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (cpu omp+tiled) vs AVG %.2fms (gpu tiled incl. transfers)\n", ms_cpu, ms_gpu);

    delete[] A; delete[] B; delete[] ref; delete[] out_cpu; delete[] out_gpu;
    return 0;
}
