#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

static const int N        = 1 << 24; // 16M floats
static const int NTHREADS = 8;

static void vecadd_serial(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

// TODO: implement parallel vector addition using std::thread.
// Divide [0, n) into NTHREADS equal chunks.
// Launch one std::thread per chunk; each thread computes c[lo..hi) = a[lo..hi) + b[lo..hi).
// Join all threads before returning.
static void vecadd_threads(const float* a, const float* b, float* c, int n) {
    // TODO
}

int main() {
    float* a     = new float[N];
    float* b     = new float[N];
    float* c_ref = new float[N];
    float* c_thr = new float[N];
    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(N - i); }

    vecadd_serial (a, b, c_ref, N);
    vecadd_threads(a, b, c_thr, N);

    printf("TEST01: std::thread vector addition correctness\n");
    bool ok = true;
    for (int i = 0; i < N && ok; i++)
        if (fabsf(c_thr[i] - c_ref[i]) > 1e-6f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) vecadd_serial(a, b, c_ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) vecadd_threads(a, b, c_thr, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_thr = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (std::thread)\n", ms_serial, ms_thr);

    delete[] a; delete[] b; delete[] c_ref; delete[] c_thr;
    return 0;
}
