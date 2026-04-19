#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <arm_neon.h>

static const int N = 1 << 24;

static void vecadd_serial(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

static void vecadd_neon(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i <= n - 4; i += 4)
        vst1q_f32(c + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) c[i] = a[i] + b[i];
}

int main() {
    float* a      = new float[N];
    float* b      = new float[N];
    float* c_ref  = new float[N];
    float* c_neon = new float[N];
    for (int i = 0; i < N; i++) { a[i] = (float)i * 1e-6f; b[i] = (float)(N - i) * 1e-6f; }

    vecadd_serial(a, b, c_ref,  N);
    vecadd_neon  (a, b, c_neon, N);

    printf("TEST01: NEON vector addition correctness\n");
    bool ok = true;
    for (int i = 0; i < N && ok; i++)
        if (fabsf(c_neon[i] - c_ref[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) vecadd_serial(a, b, c_ref,  N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) vecadd_neon(a, b, c_neon, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_neon = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (neon)\n", ms_serial, ms_neon);

    delete[] a; delete[] b; delete[] c_ref; delete[] c_neon;
    return 0;
}
