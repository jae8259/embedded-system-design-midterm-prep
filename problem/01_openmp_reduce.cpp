#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>

static const int N = 1 << 24; // 16M floats

static float reduce_serial(const float* data, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

// TODO: implement parallel reduction using OpenMP.
// Hint: #pragma omp parallel for reduction(+:sum)
static float reduce_omp(const float* data, int n) {
    float sum = 0.f;
    // TODO
    return sum;
}

int main() {
    float* data = new float[N];
    for (int i = 0; i < N; i++) data[i] = 1.f;

    float ref = reduce_serial(data, N);
    float res = reduce_omp(data, N);

    printf("TEST01: OpenMP reduce correctness (expected=%.0f)\n", ref);
    printf("%s\n", fabsf(res - ref) / ref < 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = reduce_serial(data, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) volatile float v = reduce_omp(data, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (omp)\n", ms_serial, ms_omp);

    delete[] data;
    return 0;
}
