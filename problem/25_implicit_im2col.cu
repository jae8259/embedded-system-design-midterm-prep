#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>

// Same dimensions as P17 (im2col+GEMM on CPU)
static const int H = 64, W = 64, C = 8, C_OUT = 16, K = 3;
static const int OH = H - K + 1, OW = W - K + 1; // 62 x 62
static const int K_DIM = C * K * K;               // reduction dimension
static const int N_DIM = OH * OW;                 // output spatial positions

// ── CPU reference (direct conv, for correctness) ──────────────────────────────
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

// ── Provided: Explicit im2col kernel ─────────────────────────────────────────
// Materialises the [K_DIM][N_DIM] column matrix in global memory.
// col[((ci*K+ky)*K+kx)*N_DIM + oy*OW+ox] = input[(ci*H+oy+ky)*W+ox+kx]
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

// ── Provided: Simple GEMM kernel ─────────────────────────────────────────────
// out[C_OUT][N_DIM] = weights[C_OUT][K_DIM] × col[K_DIM][N_DIM]
__global__ void gemm_kernel(const float* A, const float* B, float* C_out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output spatial position
    int row = blockIdx.y;                             // output channel
    if (col >= N_DIM || row >= C_OUT) return;
    float s = 0.f;
    for (int k = 0; k < K_DIM; k++)
        s += A[row * K_DIM + k] * B[k * N_DIM + col];
    C_out[row * N_DIM + col] = s;
}

// Wrapper: explicit im2col (materialises col_buf, then GEMM)
static void conv_explicit_gpu(const float* d_in, const float* d_w,
                               float* d_out, float* d_col) {
    int total = K_DIM * N_DIM;
    im2col_kernel<<<(total + 255) / 256, 256>>>(d_in, d_col);

    dim3 block(256);
    dim3 grid((N_DIM + 255) / 256, C_OUT);
    gemm_kernel<<<grid, block>>>(d_w, d_col, d_out);
}

// ── TODO: Implicit im2col GEMM kernel ────────────────────────────────────────
// Fused single kernel — no col_buf needed.
// Each thread computes one output element: out[co][oy][ox].
//
// Hints:
//   int out_pos = blockIdx.x * blockDim.x + threadIdx.x; // in [0, N_DIM)
//   int co      = blockIdx.y;                              // output channel
//   Decode spatial: oy = out_pos / OW; ox = out_pos % OW;
//
//   Loop over the K_DIM "virtual" im2col rows (r = 0 .. K_DIM-1):
//     Decode: ci = r / (K*K); ky = (r/K)%K; kx = r%K;
//     Accumulate: sum += weights[co*K_DIM+r] * input[(ci*H+oy+ky)*W+ox+kx];
//
//   Write: output[(co*OH+oy)*OW+ox] = sum;
__global__ void conv_implicit_kernel(const float* in, const float* w, float* out) {
    // TODO
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

    // TEST01a: explicit im2col correctness
    conv_explicit_gpu(d_in, d_w, d_out, d_col);
    cudaDeviceSynchronize();
    cudaMemcpy(h_expl, d_out, szOut * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok_expl = true;
    for (size_t i = 0; i < szOut; i++)
        if (fabsf(h_expl[i] - h_ref[i]) > 1e-3f) { ok_expl = false; break; }
    printf("TEST01a: explicit im2col: %s\n", ok_expl ? "SUCCESS" : "FAIL");

    // TEST01b: implicit im2col correctness
    cudaMemset(d_out, 0, szOut * sizeof(float));
    conv_implicit_gpu(d_in, d_w, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_impl, d_out, szOut * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok_impl = true;
    for (size_t i = 0; i < szOut; i++)
        if (fabsf(h_impl[i] - h_ref[i]) > 1e-3f) { ok_impl = false; break; }
    printf("TEST01b: implicit im2col: %s\n", ok_impl ? "SUCCESS" : "FAIL");

    // TEST02: performance (note: explicit allocates col_buf = K_DIM*N_DIM*4 bytes extra)
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
