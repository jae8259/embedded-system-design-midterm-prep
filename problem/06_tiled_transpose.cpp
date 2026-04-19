#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>

static const int N    = 4096;
static const int TILE = 64;

static void transpose_naive(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[j * n + i] = in[i * n + j];
}

// TODO: implement cache-tiled transpose to improve L1/L2 reuse.
// Hint: outer loops step by TILE (i += tile, j += tile),
//       inner loops ii in [i, min(i+tile, n)), jj in [j, min(j+tile, n))
//       out[jj*n+ii] = in[ii*n+jj]
static void transpose_tiled(const float* in, float* out, int n, int tile) {
    // TODO
}

int main() {
    float* in  = new float[(size_t)N * N];
    float* ref = new float[(size_t)N * N];
    float* out = new float[(size_t)N * N];
    for (int i = 0; i < N * N; i++) in[i] = (float)i;

    transpose_naive(in, ref, N);
    transpose_tiled(in, out, N, TILE);

    printf("TEST01: cache-tiled transpose correctness\n");
    bool ok = true;
    for (int i = 0; i < N * N && ok; i++)
        if (fabsf(out[i] - ref[i]) > 1e-5f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) transpose_naive(in, ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_naive = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) transpose_tiled(in, out, N, TILE);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_tiled = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (naive) vs AVG %.2fms (tiled)\n", ms_naive, ms_tiled);

    delete[] in; delete[] ref; delete[] out;
    return 0;
}
