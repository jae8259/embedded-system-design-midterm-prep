#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

static const int M    = 512;
static const int K    = 512;
static const int NDIM = 512;
static const int TILE = 32;

// Row-major: A[M×K], B[K×N], C[M×N]
static void matmul_naive(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int l = 0; l < k; l++) sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = sum;
        }
}

// TODO: implement tiled matrix multiply with OpenMP parallelism.
// Hints:
//   memset(C, 0, m*n*sizeof(float));
//   #pragma omp parallel for collapse(2) schedule(static)
//   Outer tile loops: for i in [0,m) step TILE, for j in [0,n) step TILE
//   Inner sequential: for l in [0,k) step TILE
//   Innermost 3 loops: ii, jj, ll with min(i+TILE,m), min(j+TILE,n), min(l+TILE,k)
//   C[ii*n+jj] += A[ii*k+ll] * B[ll*n+jj]
static void matmul_omp(const float* A, const float* B, float* C, int m, int k, int n) {
    // TODO
}

int main() {
    float* A   = new float[M * K];
    float* B   = new float[K * NDIM];
    float* ref = new float[M * NDIM];
    float* out = new float[M * NDIM];
    for (int i = 0; i < M * K; i++) A[i] = 1.f;
    for (int i = 0; i < K * NDIM; i++) B[i] = 1.f;

    matmul_naive(A, B, ref, M, K, NDIM);
    matmul_omp(A, B, out, M, K, NDIM);

    printf("TEST01: OpenMP tiled matmul correctness\n");
    bool ok = true;
    for (int i = 0; i < M * NDIM && ok; i++)
        if (fabsf(out[i] - ref[i]) > 1e-2f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_naive(A, B, ref, M, K, NDIM);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_naive = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) matmul_omp(A, B, out, M, K, NDIM);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (omp+tiled)\n", ms_naive, ms_omp);

    delete[] A; delete[] B; delete[] ref; delete[] out;
    return 0;
}
