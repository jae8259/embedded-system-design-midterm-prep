#include <cstring>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static const int M = 512, K_SZ = 512, N = 512, NTHREADS = 8;

static void matmul_ijk(const float* A, const float* B, float* C, int m, int k, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int kk = 0; kk < k; kk++)
                C[i * n + j] += A[i * k + kk] * B[kk * n + j];
}

// TODO: implement kij loop order.
// Move the k-loop outermost: for k, for i, for j.
// For each (k,i): scalar a = A[i][k], then inner j-loop: C[i][j] += a * B[k][j].
// B[k][j] is now accessed row-sequentially (cache-friendly).
static void matmul_kij(const float* A, const float* B, float* C, int m, int k, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    // TODO
}

// TODO: parallelize matmul_kij using std::thread.
// Partition the i-loop across NTHREADS threads: thread t handles rows [lo, hi).
// Each thread writes to a distinct row range of C — no races, no mutex needed.
// (Contrast: partitioning the k-loop would cause races on every C[i][j].)
static void matmul_kij_thread(const float* A, const float* B, float* C, int m, int k, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    // TODO
}

int main() {
    size_t szA = (size_t)M * K_SZ, szB = (size_t)K_SZ * N, szC = (size_t)M * N;
    float* A  = new float[szA];
    float* B  = new float[szB];
    float* C1 = new float[szC];
    float* C2 = new float[szC];
    float* C3 = new float[szC];

    for (size_t i = 0; i < szA; i++) A[i] = (float)(i % 7) / 7.f;
    for (size_t i = 0; i < szB; i++) B[i] = (float)(i % 11) / 11.f;

    matmul_ijk(A, B, C1, M, K_SZ, N);
    matmul_kij(A, B, C2, M, K_SZ, N);
    matmul_kij_thread(A, B, C3, M, K_SZ, N);

    // TEST01
    bool ok2 = true, ok3 = true;
    for (size_t i = 0; i < szC; i++) {
        if (ok2 && __builtin_fabsf(C2[i] - C1[i]) > 1e-2f) ok2 = false;
        if (ok3 && __builtin_fabsf(C3[i] - C1[i]) > 1e-2f) ok3 = false;
    }
    printf("TEST01 kij:        %s\n", ok2 ? "SUCCESS" : "FAIL");
    printf("TEST01 kij+thread: %s\n", ok3 ? "SUCCESS" : "FAIL");

    // TEST02
    double sum1 = 0, sum2 = 0, sum3 = 0;
    for (int r = 0; r < 10; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_ijk(A, B, C1, M, K_SZ, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul_kij(A, B, C2, M, K_SZ, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        matmul_kij_thread(A, B, C3, M, K_SZ, N);
        auto t3 = std::chrono::high_resolution_clock::now();
        sum1 += std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum2 += std::chrono::duration<double, std::milli>(t2 - t1).count();
        sum3 += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }
    printf("TEST02 AVG %.2fms (ijk) vs AVG %.2fms (kij) vs AVG %.2fms (kij+thread)\n",
           sum1 / 10, sum2 / 10, sum3 / 10);

    delete[] A; delete[] B; delete[] C1; delete[] C2; delete[] C3;
    return 0;
}
