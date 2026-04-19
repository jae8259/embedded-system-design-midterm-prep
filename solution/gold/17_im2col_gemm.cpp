#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

static const int H = 64, W = 64, C = 8, C_OUT = 16, K = 3;
static const int OH = H - K + 1, OW = W - K + 1;

static void conv_direct(const float* input, const float* kernel, float* output) {
    for (int co = 0; co < C_OUT; co++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                float sum = 0.f;
                for (int ci = 0; ci < C; ci++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++)
                            sum += kernel[((co*C+ci)*K+ky)*K+kx]
                                 * input[(ci*H+oy+ky)*W+ox+kx];
                output[(co*OH+oy)*OW+ox] = sum;
            }
}

static void gemm(const float* A, const float* B, float* Cout, int M, int KK, int N) {
    memset(Cout, 0, (size_t)M * N * sizeof(float));
    for (int k = 0; k < KK; k++)
        for (int i = 0; i < M; i++) {
            float a = A[i * KK + k];
            for (int j = 0; j < N; j++) Cout[i * N + j] += a * B[k * N + j];
        }
}

static void im2col(const float* input, float* col_buf) {
    for (int ci = 0; ci < C; ci++)
        for (int ky = 0; ky < K; ky++)
            for (int kx = 0; kx < K; kx++)
                for (int oy = 0; oy < OH; oy++)
                    for (int ox = 0; ox < OW; ox++)
                        col_buf[((ci*K+ky)*K+kx)*OH*OW + oy*OW+ox]
                            = input[(ci*H+oy+ky)*W+ox+kx];
}

static void conv_im2col(const float* input, const float* kernel, float* output, float* col_buf) {
    im2col(input, col_buf);
    gemm(kernel, col_buf, output, C_OUT, C * K * K, OH * OW);
}

int main() {
    int input_sz  = C * H * W;
    int kernel_sz = C_OUT * C * K * K;
    int output_sz = C_OUT * OH * OW;
    int col_sz    = C * K * K * OH * OW;

    float* input     = (float*)malloc(input_sz  * sizeof(float));
    float* kernel    = (float*)malloc(kernel_sz * sizeof(float));
    float* out_ref   = (float*)malloc(output_sz * sizeof(float));
    float* out_im2col= (float*)malloc(output_sz * sizeof(float));
    float* col_buf   = (float*)malloc(col_sz    * sizeof(float));

    for (int i = 0; i < input_sz;  i++) input[i]  = (float)(i % 17) / 17.f;
    for (int i = 0; i < kernel_sz; i++) kernel[i] = (float)(i % 13) / 13.f;

    conv_direct(input, kernel, out_ref);
    conv_im2col(input, kernel, out_im2col, col_buf);

    bool pass = true;
    for (int i = 0; i < output_sz; i++)
        if (fabsf(out_ref[i] - out_im2col[i]) > 1e-3f) { pass = false; break; }
    printf("TEST01: %s\n", pass ? "SUCCESS" : "FAIL");

    double sum_direct = 0.0, sum_im2col = 0.0;
    for (int r = 0; r < 10; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        conv_direct(input, kernel, out_ref);
        auto t1 = std::chrono::high_resolution_clock::now();
        sum_direct += std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto t2 = std::chrono::high_resolution_clock::now();
        conv_im2col(input, kernel, out_im2col, col_buf);
        auto t3 = std::chrono::high_resolution_clock::now();
        sum_im2col += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }
    printf("TEST02: AVG %.3fms (direct) vs AVG %.3fms (im2col+gemm)\n",
           sum_direct / 10.0, sum_im2col / 10.0);

    free(input); free(kernel); free(out_ref); free(out_im2col); free(col_buf);
    return 0;
}
