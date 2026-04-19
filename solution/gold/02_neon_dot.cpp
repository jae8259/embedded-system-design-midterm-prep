#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static const int N = 1 << 24;

static float dot_serial(const float* a, const float* b, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float dot_neon(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.f);
    int i = 0;
    for (; i <= n - 4; i += 4)
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

int main() {
    float* a = new float[N];
    float* b = new float[N];
    for (int i = 0; i < N; i++) { a[i] = 1.f; b[i] = 2.f; }

    float ref = dot_serial(a, b, N);
    float res = dot_neon(a, b, N);

    printf("TEST01: NEON dot product correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(res - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = dot_serial(a, b, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = dot_neon(a, b, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_neon = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (neon)\n", ms_serial, ms_neon);

    delete[] a; delete[] b;
    return 0;
}
