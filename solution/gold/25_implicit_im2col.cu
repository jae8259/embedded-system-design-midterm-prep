#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>

static const int H = 64, W = 64, C = 8, C_OUT = 16, K = 3;
static const int OH = H - K + 1, OW = W - K + 1;
static const int K_DIM = C * K * K;
static const int N_DIM = OH * OW;

static void conv_cpu(const float* in, const float* w, float* out) {
    for (int co = 0; co < C_OUT; co++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                float s = 0.f;
                for (int ci = 0; ci < C; ci++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++)
                            s += w[((co*C+ci)*K+ky)*K+kx] * in[(ci*H+oy+ky)*W+ox+kx];
                out[(co*OH+oy)*OW+ox] = s;
            }
}

__global__ void im2col_kernel(const float* in, float* col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K_DIM * N_DIM;
    if (idx >= total) return;

    int out_pos = idx % N_DIM;
    int row     = idx / N_DIM;
    int ox = out_pos % OW;
    int oy = out_pos / OW;
    int ci = row / (K * K);
    int ky = (row / K) % K;
    int kx = row % K;

    col[row * N_DIM + out_pos] = in[(ci * H + oy + ky) * W + ox + kx];
}

__global__ void gemm_kernel(const float* A, const float* B, float* C_out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= N_DIM || row >= C_OUT) return;
    float s = 0.f;
    for (int k = 0; k < K_DIM; k++)
        s += A[row * K_DIM + k] * B[k * N_DIM + col];
    C_out[row * N_DIM + col] = s;
}

static void conv_explicit_gpu(const float* d_in, const float* d_w,
                               float* d_out, float* d_col) {
    int total = K_DIM * N_DIM;
    im2col_kernel<<<(total + 255) / 256, 256>>>(d_in, d_col);
    dim3 block(256);
    dim3 grid((N_DIM + 255) / 256, C_OUT);
    gemm_kernel<<<grid, block>>>(d_w, d_col, d_out);
}

// Each thread computes one output element by decoding (ci,ky,kx) from the
// "virtual" im2col row index on-the-fly — no col_buf needed.
__global__ void conv_implicit_kernel(const float* in, const float* w, float* out) {
    int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int co      = blockIdx.y;
    if (out_pos >= N_DIM || co >= C_OUT) return;

    int oy = out_pos / OW;
    int ox = out_pos % OW;

    float s = 0.f;
    for (int r = 0; r < K_DIM; r++) {
        int ci = r / (K * K);
        int ky = (r / K) % K;
        int kx = r % K;
        s += w[co * K_DIM + r] * in[(ci * H + oy + ky) * W + ox + kx];
    }
    out[(co * OH + oy) * OW + ox] = s;
}

static void conv_implicit_gpu(const float* d_in, const float* d_w, float* d_out) {
    dim3 block(256);
    dim3 grid((N_DIM + 255) / 256, C_OUT);
    conv_implicit_kernel<<<grid, block>>>(d_in, d_w, d_out);
}

int main() {
    size_t szIn  = (size_t)C * H * W;
    size_t szW   = (size_t)C_OUT * C * K * K;
    size_t szOut = (size_t)C_OUT * OH * OW;
    size_t szCol = (size_t)K_DIM * N_DIM;

    float* h_in   = new float[szIn];
    float* h_w    = new float[szW];
    float* h_ref  = new float[szOut];
    float* h_expl = new float[szOut];
    float* h_impl = new float[szOut];

    for (size_t i = 0; i < szIn; i++) h_in[i] = (float)(i % 255) / 255.f;
    for (size_t i = 0; i < szW;  i++) h_w[i]  = 1.f / K_DIM;

    conv_cpu(h_in, h_w, h_ref);

    float *d_in, *d_w, *d_out, *d_col;
    cudaMalloc(&d_in,  szIn  * sizeof(float));
    cudaMalloc(&d_w,   szW   * sizeof(float));
    cudaMalloc(&d_out, szOut * sizeof(float));
    cudaMalloc(&d_col, szCol * sizeof(float));
    cudaMemcpy(d_in, h_in, szIn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,  h_w,  szW  * sizeof(float), cudaMemcpyHostToDevice);

    conv_explicit_gpu(d_in, d_w, d_out, d_col);
    cudaDeviceSynchronize();
    cudaMemcpy(h_expl, d_out, szOut * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok_expl = true;
    for (size_t i = 0; i < szOut; i++)
        if (fabsf(h_expl[i] - h_ref[i]) > 1e-3f) { ok_expl = false; break; }
    printf("TEST01a: explicit im2col: %s\n", ok_expl ? "SUCCESS" : "FAIL");

    cudaMemset(d_out, 0, szOut * sizeof(float));
    conv_implicit_gpu(d_in, d_w, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_impl, d_out, szOut * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok_impl = true;
    for (size_t i = 0; i < szOut; i++)
        if (fabsf(h_impl[i] - h_ref[i]) > 1e-3f) { ok_impl = false; break; }
    printf("TEST01b: implicit im2col: %s\n", ok_impl ? "SUCCESS" : "FAIL");

    const int RUNS = 100;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    for (int r = 0; r < RUNS; r++) conv_explicit_gpu(d_in, d_w, d_out, d_col);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_expl; cudaEventElapsedTime(&ms_expl, s, e); ms_expl /= RUNS;

    cudaEventRecord(s);
    for (int r = 0; r < RUNS; r++) conv_implicit_gpu(d_in, d_w, d_out);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_impl; cudaEventElapsedTime(&ms_impl, s, e); ms_impl /= RUNS;

    printf("TEST02:\nAVG %.3fms (explicit, col_buf=%zuKB) vs AVG %.3fms (implicit, no buf)\n",
           ms_expl, szCol * sizeof(float) / 1024, ms_impl);

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out); cudaFree(d_col);
    delete[] h_in; delete[] h_w; delete[] h_ref; delete[] h_expl; delete[] h_impl;
    return 0;
}
