#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cfloat>

static const int N = 2048;
static const int D = 32;

static void cpu_attention(const float* Q, const float* K, const float* V, float* O) {
    for (int i = 0; i < N; i++) {
        float m = -FLT_MAX;
        float l = 0.0f;
        float o_tmp[D] = {0.0f};

        for (int j = 0; j <= i; j++) {
            float dot = 0.0f;
            for (int k = 0; k < D; k++)
                dot += Q[i * D + k] * K[j * D + k];

            float m_new = std::max(m, dot);
            float rescale = std::exp(m - m_new);
            float p = std::exp(dot - m_new);

            l = l * rescale + p;
            for (int k = 0; k < D; k++)
                o_tmp[k] = o_tmp[k] * rescale + p * V[j * D + k];
            m = m_new;
        }
        for (int k = 0; k < D; k++)
            O[i * D + k] = o_tmp[k] / l;
    }
}

// Butterfly all-reduce: every lane gets the sum across all 32 lanes.
__device__ float warp_all_sum(float val) {
    unsigned const mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(mask, val, offset);
    return val;
}

__global__ void warp_flash_attention(const float* Q, const float* K, const float* V, float* O) {
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    int q_idx = blockIdx.x * (blockDim.x / 32) + wid;
    if (q_idx >= N) return;

    // Each lane loads its own dimension of the Q row.
    float q_val = Q[q_idx * D + lane];

    float m_i  = -FLT_MAX;
    float l_i  = 0.0f;
    float o_val = 0.0f;

    for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
        // Dot product: each lane holds one dimension; sum across the warp.
        float dot = warp_all_sum(q_val * K[k_idx * D + lane]);

        float m_new   = fmaxf(m_i, dot);
        float rescale = expf(m_i - m_new);
        float p       = expf(dot - m_new);

        l_i   = l_i * rescale + p;
        o_val = o_val * rescale + p * V[k_idx * D + lane];
        m_i   = m_new;
    }

    O[q_idx * D + lane] = o_val / l_i;
}

int main() {
    size_t bytes = N * D * sizeof(float);
    float *h_Q     = new float[N * D];
    float *h_K     = new float[N * D];
    float *h_V     = new float[N * D];
    float *h_O_ref = new float[N * D];
    float *h_O_gpu = new float[N * D];

    for (int i = 0; i < N * D; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f;
        h_K[i] = (float)(rand() % 100) / 100.0f;
        h_V[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes); cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes); cudaMalloc(&d_O, bytes);
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_attention(h_Q, h_K, h_V, h_O_ref);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;

    int threadsPerBlock = 256;
    int warpsPerBlock   = threadsPerBlock / 32;
    int blocks          = (N + warpsPerBlock - 1) / warpsPerBlock;

    cudaMemset(d_O, 0, bytes);
    warp_flash_attention<<<blocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O_gpu, d_O, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float err = fabsf(h_O_ref[i] - h_O_gpu[i]);
        if (err > max_err) max_err = err;
    }
    printf("TEST 01: Correctness Check\n");
    printf("Max Error = %e\n", max_err);
    printf("%s\n\n", max_err < 1e-3f ? "SUCCESS" : "FAIL");

    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    cudaEventRecord(ev_s);
    for (int i = 0; i < 100; i++)
        warp_flash_attention<<<blocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O);
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_gpu; cudaEventElapsedTime(&ms_gpu, ev_s, ev_e); ms_gpu /= 100.0f;

    printf("TEST 02: Performance\n");
    printf("CPU Time : %8.3f ms\n", diff_cpu.count() * 1000.0f);
    printf("GPU Time : %8.3f ms\n", ms_gpu);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O_ref; delete[] h_O_gpu;
    return 0;
}
