#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;

constexpr int THREADS_X = BLOCK_N / THREAD_N;
constexpr int THREADS_Y = BLOCK_M / THREAD_M;

static_assert(BLOCK_M % THREAD_M == 0, "BLOCK_M must divide THREAD_M");
static_assert(BLOCK_N % THREAD_N == 0, "BLOCK_N must divide THREAD_N");

constexpr int M = 257;
constexpr int K = 255;
constexpr int N = 259;

static void matmul_serial(const float *A, const float *B, float *C, int m,
                          int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.f;
      for (int kk = 0; kk < k; ++kk)
        sum += A[i * k + kk] * B[kk * n + j];
      C[i * n + j] = sum;
    }
  }
}
__global__ void matmul_tiled_regtile_kernel(const float *A, // M x K
                                            const float *B, // K x N
                                            float *C,       // M x N
                                            int M, int K, int N) {
  __shared__ float As[BLOCK_M][BLOCK_K];
  __shared__ float Bs[BLOCK_K][BLOCK_N];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int block_row = blockIdx.y * BLOCK_M;
  const int block_col = blockIdx.x * BLOCK_N;

  const int thread_row_base = ty * THREAD_M;
  const int thread_col_base = tx * THREAD_N;

  float acc[THREAD_M][THREAD_N];

#pragma unroll
  for (int i = 0; i < THREAD_M; ++i) {
#pragma unroll
    for (int j = 0; j < THREAD_N; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  // packing
  const int linear_tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;

  for (int kb = 0; kb < K; kb += BLOCK_K) {
    // Load A tile: BLOCK_M x BLOCK_K
    for (int idx = linear_tid; idx < BLOCK_M * BLOCK_K; idx += num_threads) {
      const int r = idx / BLOCK_K;
      const int c = idx % BLOCK_K;

      const int global_row = block_row + r;
      const int global_col = kb + c;

      As[r][c] = A[global_row * K + global_col];
    }

    // Load B tile: BLOCK_K x BLOCK_N
    for (int idx = linear_tid; idx < BLOCK_K * BLOCK_N; idx += num_threads) {
      const int r = idx / BLOCK_N;
      const int c = idx % BLOCK_N;

      const int global_row = kb + r;
      const int global_col = block_col + c;

      Bs[r][c] = B[global_row * N + global_col];
    }

    __syncthreads();

#pragma unroll
    for (int k_inner = 0; k_inner < BLOCK_K; ++k_inner) {
      float a_frag[THREAD_M];
      float b_frag[THREAD_N];

#pragma unroll
      for (int i = 0; i < THREAD_M; ++i) {
        a_frag[i] = As[thread_row_base + i][k_inner];
      }

#pragma unroll
      for (int j = 0; j < THREAD_N; ++j) {
        b_frag[j] = Bs[k_inner][thread_col_base + j];
      }

#pragma unroll
      for (int i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < THREAD_M; ++i) {
    const int global_row = block_row + thread_row_base + i;
    if (global_row < M) {
#pragma unroll
      for (int j = 0; j < THREAD_N; ++j) {
        const int global_col = block_col + thread_col_base + j;
        if (global_col < N) {
          C[global_row * N + global_col] = acc[i][j];
        }
      }
    }
  }
}

__global__ void matmul_kernel(const float *A, const float *B, float *C, int m,
                              int k, int n) {
  __shared__ float As[BLOCK_M][BLOCK_K];
  __shared__ float Bs[BLOCK_K][BLOCK_N];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int linear_tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;

  const int block_row = blockIdx.y * BLOCK_M;
  const int block_col = blockIdx.x * BLOCK_N;
  const int thread_row_base = ty * THREAD_M;
  const int thread_col_base = tx * THREAD_N;

  float acc[THREAD_M][THREAD_N];
#pragma unroll
  for (int i = 0; i < THREAD_M; ++i)
#pragma unroll
    for (int j = 0; j < THREAD_N; ++j)
      acc[i][j] = 0.f;

  for (int kb = 0; kb < k; kb += BLOCK_K) {
    for (int idx = linear_tid; idx < BLOCK_M * BLOCK_K; idx += num_threads) {
      const int r = idx / BLOCK_K;
      const int c = idx % BLOCK_K;
      const int global_row = block_row + r;
      const int global_col = kb + c;
      As[r][c] = (global_row < m && global_col < k)
                     ? A[global_row * k + global_col]
                     : 0.f;
    }

    for (int idx = linear_tid; idx < BLOCK_K * BLOCK_N; idx += num_threads) {
      const int r = idx / BLOCK_N;
      const int c = idx % BLOCK_N;
      const int global_row = kb + r;
      const int global_col = block_col + c;
      Bs[r][c] = (global_row < k && global_col < n)
                     ? B[global_row * n + global_col]
                     : 0.f;
    }

    __syncthreads();

#pragma unroll
    for (int k_inner = 0; k_inner < BLOCK_K; ++k_inner) {
      float a_frag[THREAD_M];
      float b_frag[THREAD_N];

#pragma unroll
      for (int i = 0; i < THREAD_M; ++i)
        a_frag[i] = As[thread_row_base + i][k_inner];

#pragma unroll
      for (int j = 0; j < THREAD_N; ++j)
        b_frag[j] = Bs[k_inner][thread_col_base + j];

#pragma unroll
      for (int i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < THREAD_M; ++i) {
    const int global_row = block_row + thread_row_base + i;
    if (global_row < m) {
#pragma unroll
      for (int j = 0; j < THREAD_N; ++j) {
        const int global_col = block_col + thread_col_base + j;
        if (global_col < n)
          C[global_row * n + global_col] = acc[i][j];
      }
    }
  }
}

} // namespace

void matmul_cuda(const float *d_A, const float *d_B, float *d_C, int m, int k,
                 int n) {
  dim3 block(THREADS_X, THREADS_Y);
  dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);
  matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
}

static void matmul_cuda_host(const float *A, const float *B, float *C, int m,
                             int k, int n) {
  const size_t a_bytes = (size_t)m * k * sizeof(float);
  const size_t b_bytes = (size_t)k * n * sizeof(float);
  const size_t c_bytes = (size_t)m * n * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, a_bytes);
  cudaMalloc(&d_B, b_bytes);
  cudaMalloc(&d_C, c_bytes);

  cudaMemcpy(d_A, A, a_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, b_bytes, cudaMemcpyHostToDevice);

  matmul_cuda(d_A, d_B, d_C, m, k, n);
  cudaDeviceSynchronize();

  cudaMemcpy(C, d_C, c_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C_ref = new float[M * N];
  float *C_gpu = new float[M * N];

  for (int i = 0; i < M * K; ++i)
    A[i] = float((i % 13) - 6) * 0.25f;
  for (int i = 0; i < K * N; ++i)
    B[i] = float((i % 17) - 8) * 0.125f;

  matmul_serial(A, B, C_ref, M, K, N);
  matmul_cuda_host(A, B, C_gpu, M, K, N);

  bool ok = true;
  for (int i = 0; i < M * N; ++i) {
    if (fabsf(C_ref[i] - C_gpu[i]) > 1e-2f) {
      ok = false;
      break;
    }
  }

  printf("TEST01: CUDA blocked matmul correctness\n");
  printf("%s\n", ok ? "SUCCESS" : "FAIL");

  const int RUNS = 10;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUNS; ++i)
    matmul_serial(A, B, C_ref, M, K, N);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms_serial =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUNS; ++i)
    matmul_cuda_host(A, B, C_gpu, M, K, N);
  t1 = std::chrono::high_resolution_clock::now();
  double ms_cuda =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;

  printf("TEST02:\nAVG %.2fms (serial) vs AVG %.2fms (cuda blocked incl. "
         "transfers)\n",
         ms_serial, ms_cuda);

  delete[] A;
  delete[] B;
  delete[] C_ref;
  delete[] C_gpu;
  return 0;
}
