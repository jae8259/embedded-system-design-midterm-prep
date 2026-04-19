#include <cstdio>
#include <chrono>
#include <omp.h>

static const int C = 64, H = 128, W = 128, K = 3;
static const int OH = H - K + 1, OW = W - K + 1;

static void dwconv_serial(const float* input, const float* kernels, float* output) {
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                float sum = 0.f;
                for (int ky = 0; ky < K; ky++)
                    for (int kx = 0; kx < K; kx++)
                        sum += kernels[(c * K + ky) * K + kx]
                             * input[(c * H + oy + ky) * W + ox + kx];
                output[(c * OH + oy) * OW + ox] = sum;
            }
}

static void dwconv_omp(const float* input, const float* kernels, float* output) {
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                float sum = 0.f;
                for (int ky = 0; ky < K; ky++)
                    for (int kx = 0; kx < K; kx++)
                        sum += kernels[(c * K + ky) * K + kx]
                             * input[(c * H + oy + ky) * W + ox + kx];
                output[(c * OH + oy) * OW + ox] = sum;
            }
}

int main() {
    size_t szIn  = (size_t)C * H * W;
    size_t szK   = (size_t)C * K * K;
    size_t szOut = (size_t)C * OH * OW;

    float* input   = new float[szIn];
    float* kernels = new float[szK];
    float* out_ref = new float[szOut];
    float* out_omp = new float[szOut];

    for (size_t i = 0; i < szIn; i++) input[i]   = (float)(i % 255) / 255.f;
    for (size_t i = 0; i < szK;  i++) kernels[i] = 1.f / (K * K);

    dwconv_serial(input, kernels, out_ref);
    dwconv_omp(input, kernels, out_omp);

    // TEST01
    bool ok = true;
    for (size_t i = 0; i < szOut; i++)
        if (__builtin_fabsf(out_omp[i] - out_ref[i]) > 1e-4f) { ok = false; break; }
    printf("TEST01: %s\n", ok ? "SUCCESS" : "FAIL");

    // TEST02
    double sum1 = 0, sum2 = 0;
    for (int r = 0; r < 10; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dwconv_serial(input, kernels, out_ref);
        auto t1 = std::chrono::high_resolution_clock::now();
        dwconv_omp(input, kernels, out_omp);
        auto t2 = std::chrono::high_resolution_clock::now();
        sum1 += std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum2 += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    printf("TEST02 AVG %.2fms (serial) vs AVG %.2fms (omp)\n",
           sum1 / 10, sum2 / 10);

    delete[] input; delete[] kernels; delete[] out_ref; delete[] out_omp;
    return 0;
}
