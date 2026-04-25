#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cfloat>

static const int N = 2048; // 시퀀스 길이 (문장 토큰 개수)
static const int D = 32;   // Head Dimension (워프 크기와 동일하게 맞춰 최적화)

// =====================================================================
// CPU Baseline: 정석적인 Attention 계산 (3-Pass)
// =====================================================================
static void cpu_attention(const float* Q, const float* K, const float* V, float* O) {  
    for (int i = 0; i < N; i++) {
        float m = -FLT_MAX;
        float l = 0.0f;
        float o_tmp[D] = {0.0f}; // 각 차원별 결과를 저장할 임시 배열

        for (int j = 0; j <= i; j++) {
            // 1. Q * K^T 내적 (Dot Product)
            float dot = 0.0f;
            for (int k = 0; k < D; k++) {
                dot += Q[i * D + k] * K[j * D + k];
            }

            // 2. Online Softmax & Rescaling
            float m_new = std::max(m, dot);
            float rescale = std::exp(m - m_new);
            float p = std::exp(dot - m_new);

            l = l * rescale + p;
            
            // 3. O = P * V
            for (int k = 0; k < D; k++) {
                o_tmp[k] = o_tmp[k] * rescale + p * V[j * D + k];
            }
            m = m_new;
        }

        // 4. 최종 결과 저장
        for (int k = 0; k < D; k++) {
            O[i * D + k] = o_tmp[k] / l;
        }
    }
}

// =====================================================================
// Helper: Warp All-Reduce Sum
// =====================================================================
// 워프 내의 모든 스레드가 가진 val 값을 전부 더해서, 
// 0번 스레드만 갖는게 아니라 **32명 전원에게 동일한 총합을 반환**하는 마법의 함수입니다.
__device__ float warp_all_sum(float val) {
    // TODO 1: __shfl_xor_sync 를 사용하여 32개 스레드의 val 합치기
    // 힌트: offset을 16, 8, 4, 2, 1로 줄여나가며 xor 연산 통신을 하면, 
    // 나비(Butterfly) 모양으로 데이터가 섞이며 전원이 총합을 갖게 됩니다.
    unsigned const mask = 0xffffffff;
    
    return val;
}

// =====================================================================
// Optimized GPU: Warp-level FlashAttention (1 Warp = 1 Query Token)
// =====================================================================
__global__ void warp_flash_attention(const float* Q, const float* K, const float* V, float* O) {
    int tid = threadIdx.x;
    int lane = tid % 32;       // 내가 워프 내에서 몇 번째 스레드인가? (0~31) == 내가 맡은 차원(Dimension)
    int wid = tid / 32;        // 내가 속한 블록 내의 몇 번째 워프인가?
    
    // 글로벌 전체 기준으로 내가 맡은 Query 행(Row) 번호
    int q_idx = blockIdx.x * (blockDim.x / 32) + wid; 
    
    if (q_idx >= N) return;

    // TODO 2: 각 스레드가 Q 행렬에서 '자신의 차원(lane)'에 해당하는 값 1개만 로드
    // float q_val = Q[ ... ];
    
    
    // 로컬 누적 변수 초기화
    // 참고: m_i와 l_i는 워프 내 32명 전원이 항상 똑같은 값을 가집니다.
    // 하지만 o_val은 각 스레드(lane)마다 서로 다른 V값을 누적하므로 고유한 값을 가집니다!
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o_val = 0.0f;

    // TODO 3: 전체 Key/Value 토큰(0부터 N-1)을 순회하며 어텐션 계산
    for (int k_idx = 0; k_idx <= q_idx; k_idx++) {

    }

    // TODO 4: 계산이 끝난 O 벡터를 글로벌 메모리에 안전하게 저장
}

// =====================================================================
// Main & Benchmark
// =====================================================================
int main() {
    size_t bytes = N * D * sizeof(float);
    float *h_Q = new float[N * D];
    float *h_K = new float[N * D];
    float *h_V = new float[N * D];
    float *h_O_ref = new float[N * D];
    float *h_O_gpu = new float[N * D];

    // 랜덤 초기화
    for (int i = 0; i < N * D; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f;
        h_K[i] = (float)(rand() % 100) / 100.0f;
        h_V[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    // 1. CPU Reference 실행
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_attention(h_Q, h_K, h_V, h_O_ref);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;

    // 2. GPU 커널 실행 세팅 (블록당 256 스레드 = 8 워프)
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int blocks = (N + warpsPerBlock - 1) / warpsPerBlock; // 각 워프가 1 Row를 맡음

    cudaMemset(d_O, 0, bytes);
    warp_flash_attention<<<blocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_gpu, d_O, bytes, cudaMemcpyDeviceToHost);

    // 3. 정합성 검사 (N*D 개의 데이터 중 최대 오차 확인)
    float max_err = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float err = fabsf(h_O_ref[i] - h_O_gpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("TEST 01: Correctness Check\n");
    printf("Max Error = %e\n", max_err);
    // 허용 오차는 1e-4 내외로 설정
    printf("%s\n\n", max_err < 1e-3f ? "SUCCESS" : "FAIL");

    // 4. 성능 비교
    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    for (int i = 0; i < 100; i++) {
        warp_flash_attention<<<blocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O);
    }
    cudaEventRecord(ev_e); cudaEventSynchronize(ev_e);
    float ms_gpu; cudaEventElapsedTime(&ms_gpu, ev_s, ev_e); ms_gpu /= 100.0f;

    printf("TEST 02: Performance\n");
    printf("CPU Time : %8.3f ms\n", diff_cpu.count() * 1000.0f);
    printf("GPU Time : %8.3f ms\n", ms_gpu);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O_ref; delete[] h_O_gpu;
    return 0;
}