#include "../support/prefix_sum.h"

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

namespace {

constexpr int ThreadsPerBlock = 256;

void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) {
        return;
    }

    std::cerr << "CUDA error during " << operation << ": "
              << cudaGetErrorString(error) << '\n';
    std::exit(EXIT_FAILURE);
}

__global__ void prefix_sum_kernel(const int* input, int* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int sum = 0;
    for (int i = 0; i <= idx; ++i) {
        sum += input[i];
    }

    output[idx] = sum;
}

}  // namespace

void prefix_sum(const int* input, int* output, int n) {
    if (n <= 0) {
        return;
    }

    int* device_input = nullptr;
    int* device_output = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(int);

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), bytes),
               "cudaMalloc(device_input)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes),
               "cudaMalloc(device_output)");

    check_cuda(
        cudaMemcpy(device_input, input, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(host_to_device)");

    const int blocks = (n + ThreadsPerBlock - 1) / ThreadsPerBlock;
    prefix_sum_kernel<<<blocks, ThreadsPerBlock>>>(device_input, device_output, n);

    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    check_cuda(
        cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(device_to_host)");

    check_cuda(cudaFree(device_input), "cudaFree(device_input)");
    check_cuda(cudaFree(device_output), "cudaFree(device_output)");
}
