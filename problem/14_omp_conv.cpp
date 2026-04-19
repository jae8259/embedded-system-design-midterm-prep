#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>

static const int H = 2048, W = 2048, K = 3;
static const int OH = H - K + 1, OW = W - K + 1;

static void conv_serial(const float* img, const float* ker, float* out) {
    for (int y = 0; y < OH; y++)
        for (int x = 0; x < OW; x++) {
            float sum = 0.f;
            for (int ky = 0; ky < K; ky++)
                for (int kx = 0; kx < K; kx++)
                    sum += ker[ky * K + kx] * img[(y + ky) * W + x + kx];
            out[y * OW + x] = sum;
        }
}

// TODO: parallelize the outer pixel loops with OpenMP.
// Hint: #pragma omp parallel for collapse(2) schedule(static)
// The inner kernel loops (ky, kx) are independent per output pixel — no race conditions.
static void conv_omp(const float* img, const float* ker, float* out) {
    // TODO
}

int main() {
    float* img     = new float[H * W];
    float* ker     = new float[K * K];
    float* out_ref = new float[OH * OW];
    float* out_omp = new float[OH * OW];

    for (int i = 0; i < H * W; i++) img[i] = (float)(i % 255) / 255.f;
    for (int i = 0; i < K * K; i++) ker[i] = 1.f / (K * K);

    conv_serial(img, ker, out_ref);
    conv_omp   (img, ker, out_omp);

    printf("TEST01: OMP convolution correctness\n");
    bool ok = true;
    for (int i = 0; i < OH * OW && ok; i++)
        if (fabsf(out_omp[i] - out_ref[i]) > 1e-4f) ok = false;
    printf("%s\n", ok ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) conv_serial(img, ker, out_ref);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) conv_omp(img, ker, out_omp);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (omp)\n", ms_serial, ms_omp);

    delete[] img; delete[] ker; delete[] out_ref; delete[] out_omp;
    return 0;
}
