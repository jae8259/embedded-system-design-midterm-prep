#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <arm_neon.h>

static const int M = 512, K_SZ = 512, N = 512;

static void matmul_serial(const float* A, const float* B, float* C, int m, int k, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    for (int kk = 0; kk < k; kk++)
        for (int i = 0; i < m; i++) {
            float a = A[i * k + kk];
            for (int j = 0; j < n; j++) C[i * n + j] += a * B[kk * n + j];
        }
}

// TODO: implement NEON matmul using vmlaq_f32.
// Use kij loop order. For each (k,i): broadcast A[i][k] with vdupq_n_f32,
// then for j = 0,4,8,...:
//   load c_vec = vld1q_f32(C+i*n+j)
//   c_vec = vmlaq_f32(c_vec, a_vec, vld1q_f32(B+kk*n+j))
//   vst1q_f32(C+i*n+j, c_vec)
// Requires n % 4 == 0.
static void matmul_neon(const float* A, const float* B, float* C, int m, int k, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    // TODO
}

int main() {
    float* A     = (float*)malloc((size_t)M * K_SZ * sizeof(float));
    float* B     = (float*)malloc((size_t)K_SZ * N * sizeof(float));
    float* C_ref = (float*)malloc((size_t)M * N * sizeof(float));
    float* C_neon= (float*)malloc((size_t)M * N * sizeof(float));

    for (int i = 0; i < M * K_SZ; i++) A[i] = (float)(i % 7)  / 7.f;
    for (int i = 0; i < K_SZ * N; i++) B[i] = (float)(i % 11) / 11.f;

    matmul_serial(A, B, C_ref,  M, K_SZ, N);
    matmul_neon  (A, B, C_neon, M, K_SZ, N);

    bool pass = true;
    for (int i = 0; i < M * N; i++)
        if (fabsf(C_ref[i] - C_neon[i]) > 1e-2f) { pass = false; break; }
    printf("TEST01: %s\n", pass ? "SUCCESS" : "FAIL");

    double sum_serial = 0.0, sum_neon = 0.0;
    for (int r = 0; r < 10; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_serial(A, B, C_ref,  M, K_SZ, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        sum_serial += std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto t2 = std::chrono::high_resolution_clock::now();
        matmul_neon(A, B, C_neon, M, K_SZ, N);
        auto t3 = std::chrono::high_resolution_clock::now();
        sum_neon += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }
    printf("TEST02: AVG %.3fms (serial) vs AVG %.3fms (neon)\n",
           sum_serial / 10.0, sum_neon / 10.0);

    free(A); free(B); free(C_ref); free(C_neon);
    return 0;
}
