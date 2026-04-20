#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>

constexpr int kBlockSize = 512;
constexpr int kNumBlocks = 256;

// ── DEMO: warp shuffle reduction ─────────────────────────────────────────────
// Provided: sums v across 32 lanes using __shfl_down_sync.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// ── TODO 1: reduce_kernel ─────────────────────────────────────────────────────
// Each block reduces a portion of in[] into a single partial sum in out[blockIdx.x].
// Steps:
//   1. Grid-stride loop: each thread sums its assigned elements into a local float `sum`.
//        int idx = blockIdx.x * blockDim.x + threadIdx.x;
//        int stride = blockDim.x * gridDim.x;
//        for (int i = idx; i < n; i += stride) sum += in[i];
//   2. Store sum into shared memory: sdata[tid] = sum; __syncthreads();
//   3. Tree reduction in shared mem down to 32 elements:
//        for (int s = blockDim.x/2; s > 32; s >>= 1) {
//            if (tid < s) sdata[tid] += sdata[tid + s];
//            __syncthreads(); }
//   4. Warp-level final: if (tid < 32) { float v = sdata[tid] + sdata[tid+32]; v = warp_reduce_sum(v); if (tid==0) out[blockIdx.x] = v; }
__global__ void reduce_kernel(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    // TODO
}

// ── TODO 2: reduce (host wrapper) ─────────────────────────────────────────────
// Launch reduce_kernel to compute the total sum of g_idata[0..n-1].
// Result must end up in g_odata[0] after this call.
// Hints:
//   size_t shm_size = kBlockSize * sizeof(float);
//   int num_blocks = std::min((n + kBlockSize - 1) / kBlockSize, kNumBlocks);
//   reduce_kernel<<<num_blocks, kBlockSize, shm_size>>>(g_idata, g_odata, n);
//   If num_blocks > 1, launch a second pass to reduce the partial sums:
//     reduce_kernel<<<1, kBlockSize, shm_size>>>(g_odata, g_idata, num_blocks);
//     cudaMemcpy(g_odata, g_idata, sizeof(float), cudaMemcpyDeviceToDevice);
void reduce(float* h_idata, float* h_odata, float* g_idata, float* g_odata, int n) {
    (void)h_idata;
    (void)h_odata;
    if (n <= 0) return;
    // TODO
}

// ── Serial reference ──────────────────────────────────────────────────────────
static float reduce_serial(const float* data, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += data[i];
    return s;
}

int main() {
    const int N = 1 << 24; // 16M floats

    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f; // expected sum = N

    float* g_in;  cudaMalloc(&g_in,  (size_t)N         * sizeof(float));
    float* g_out; cudaMalloc(&g_out, (size_t)kNumBlocks * sizeof(float));
    cudaMemcpy(g_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    float ref = reduce_serial(h_in, N);
    reduce(nullptr, nullptr, g_in, g_out, N);
    cudaDeviceSynchronize();

    float cuda_sum;
    cudaMemcpy(&cuda_sum, g_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("TEST01: serial=%.0f cuda=%.0f %s\n",
           ref, cuda_sum,
           fabsf(cuda_sum - ref) < ref * 1e-3f ? "SUCCESS" : "FAIL");

    const int RUNS = 10;
    // Serial timing
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) { volatile float s = reduce_serial(h_in, N); (void)s; }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    // CUDA timing
    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) {
        reduce(nullptr, nullptr, g_in, g_out, N);
        cudaDeviceSynchronize();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_cuda = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

    printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (cuda)\n", ms_serial, ms_cuda);

    cudaFree(g_in); cudaFree(g_out);
    delete[] h_in;
    return 0;
}
