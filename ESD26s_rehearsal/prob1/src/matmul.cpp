#include <stdlib.h>
#include <iostream>
#include <arm_neon.h>
#include <pthread.h>
#include <thread>
#include <omp.h>
#include "../support/matmul.h"

void matmul(const int *inputA, const int *inputB, int *output, const int M, const int N, const int K) {
    
#pragma omp parallel for collapse(2)

for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
        
            for (int k = 0; k < K; k++) {
                output[i * N + j] += inputA[i * K + k] * inputB[k * N + j];
            }
        }
    }
    return;
}