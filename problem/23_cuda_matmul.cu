#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>

namespace {

constexpr int BLOCK_M  = 64;
constexpr int BLOCK_N  = 64;
constexpr int BLOCK_K  = 16;

constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;

constexpr int THREADS_X = BLOCK_N / THREAD_N; // 16
constexpr int THREADS_Y = BLOCK_M / THREAD_M; // 16

static_assert(BLOCK_M % THREAD_M == 0, "BLOCK_M must divide THREAD_M");
static_assert(BLOCK_N % THREAD_N == 0, "BLOCK_N must divide THREAD_N");

constexpr int M = 257;
constexpr int K = 255;
constexpr int N = 259;

static void matmul_serial(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int kk = 0; kk < k; ++kk) sum += A[i * k + kk] * B[kk * n + j];
            C[i * n + j] = sum;
        }
    }
}

// TODO: implement a tiled CUDA matmul kernel.
// Each block computes a BLOCK_M x BLOCK_N output tile.
// Each thread computes a THREAD_M x THREAD_N register tile.
//
// Hints:
//   - Use shared memory:
//       __shared__ float As[BLOCK_M][BLOCK_K];
//       __shared__ float Bs[BLOCK_K][BLOCK_N];
//   - block_row = blockIdx.y * BLOCK_M
//     block_col = blockIdx.x * BLOCK_N
//   - thread_row_base = threadIdx.y * THREAD_M
//     thread_col_base = threadIdx.x * THREAD_N
//   - Cooperatively load A and B tiles with a linear thread id.
//   - Guard out-of-bounds global loads with 0.f.
//   - Accumulate into acc[THREAD_M][THREAD_N].
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int m, int k, int n)
{
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    // TODO
}

} // namespace

// TODO: launch matmul_kernel with:
//   dim3 block(THREADS_X, THREADS_Y);
//   dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);
void matmul_cuda(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int m, int k, int n)
{
    // TODO
}

static void matmul_cuda_host(const float* A, const float* B, float* C, int m, int k, int n) {
    const size_t a_bytes = (size_t)m * k * sizeof(float);
    const size_t b_bytes = (size_t)k * n * sizeof(float);
    const size_t c_bytes = (size_t)m * n * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, a_bytes);
    cudaMalloc(&d_B, b_bytes);
    cudaMalloc(&d_C, c_bytes);

    cudaMemcpy(d_A, A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, b_bytes, cudaMemcpyHostToDevice);

    matmul_cuda(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, c_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C_ref = new float[M * N];
    float* C_gpu = new float[M * N];

    for (int i = 0; i < M * K; ++i) A[i] = float((i % 13) - 6) * 0.25f;
    for (int i = 0; i < K * N; ++i) B[i] = float((i % 17) - 8) * 0.125f;

    matmul_serial(A, B, C_ref, M, K, N);
    matmul_cuda_host(A, B, C_gpu, M, K, N);

    bool ok = true;
    for (int i = 0; i < M * N; ++i) {
        if (fabsf(C_ref[i] - C_gpu[i]) > 1e-2f) {
            ok = false;
            break;
        }
    }

    printf("TEST01: CUDA blocked matmul correctness\n");
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; ++i) matmul_serial(A, B, C_ref, M, K, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; ++i) matmul_cuda_host(A, B, C_gpu, M, K, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_cuda = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (cuda blocked incl. transfers)\n",
           ms_serial, ms_cuda);

    delete[] A;
    delete[] B;
    delete[] C_ref;
    delete[] C_gpu;
    return 0;
}
