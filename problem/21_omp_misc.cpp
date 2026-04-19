#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <climits>
#include <cstring>
#include <omp.h>

static const int N = 1 << 22; // 4M elements

// ── DEMO 1: reduction ────────────────────────────────────────────────────────
// Provided: shows reduction(+:sum) clause.
static float sum_reduction(const float* data, int n) {
    float sum = 0.f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

// ── DEMO 2: master ───────────────────────────────────────────────────────────
// Provided: master thread prints thread count inside a parallel region.
static void master_demo() {
    #pragma omp parallel
    {
        #pragma omp master
        printf("  master demo: %d threads running, %d hw processors\n",
               omp_get_num_threads(), omp_get_num_procs());
    }
}

// ── TODO 1: private + critical ───────────────────────────────────────────────
// Find the array maximum using a parallel region.
// Each thread maintains a thread-private local_max.
// Use #pragma omp critical to safely update the global max.
// Hint:
//   int local_max = INT_MIN;
//   #pragma omp parallel private(local_max)
//   { local_max = INT_MIN;
//     #pragma omp for
//     for (int i = 0; i < n; i++) if (data[i] > local_max) local_max = data[i];
//     #pragma omp critical
//     if (local_max > *global_max) *global_max = local_max; }
static int find_max_parallel(const int* data, int n) {
    int global_max = INT_MIN;
    // TODO
    return global_max;
}

// ── TODO 2: atomic ───────────────────────────────────────────────────────────
// Build a histogram using #pragma omp atomic to avoid races on hist[].
// Hint:
//   #pragma omp parallel for schedule(static)
//   for (int i = 0; i < n; i++) {
//       #pragma omp atomic
//       hist[data[i] % BINS]++;
//   }
static const int BINS = 256;
static void histogram_serial(const int* data, int* hist, int n) {
    for (int i = 0; i < n; i++) hist[data[i] % BINS]++;
}
static void histogram_atomic(const int* data, int* hist, int n) {
    // TODO
}

int main() {
    float* fdata = new float[N];
    int*   idata = new int[N];
    for (int i = 0; i < N; i++) { fdata[i] = (float)(i % 1000); idata[i] = i; }

    // Reduction demo (provided, always works)
    float s = sum_reduction(fdata, N);
    printf("DEMO reduction:  sum=%.0f (expected %.0f)\n", s, (float)N/2.f * 999.f);

    // Master demo (provided, always works)
    master_demo();

    // TEST01a: find_max_parallel
    int ref_max = 0;
    for (int i = 0; i < N; i++) if (idata[i] > ref_max) ref_max = idata[i];
    int par_max = find_max_parallel(idata, N);
    printf("TEST01a: find_max (private+critical): expected=%d got=%d %s\n",
           ref_max, par_max, (par_max == ref_max) ? "SUCCESS" : "FAIL");

    // TEST01b: histogram_atomic
    int h_ref[BINS] = {}, h_omp[BINS] = {};
    histogram_serial(idata, h_ref, N);
    histogram_atomic(idata, h_omp, N);
    bool ok = true;
    for (int i = 0; i < BINS && ok; i++) if (h_omp[i] != h_ref[i]) ok = false;
    printf("TEST01b: histogram (atomic): %s\n", ok ? "SUCCESS" : "FAIL");

    // TEST02: benchmark histogram serial vs atomic
    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) { memset(h_ref,0,sizeof(h_ref)); histogram_serial(idata,h_ref,N); }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double,std::milli>(t1-t0).count()/RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) { memset(h_omp,0,sizeof(h_omp)); histogram_atomic(idata,h_omp,N); }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double,std::milli>(t1-t0).count()/RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (atomic)\n", ms_serial, ms_omp);

    delete[] fdata; delete[] idata;
    return 0;
}
