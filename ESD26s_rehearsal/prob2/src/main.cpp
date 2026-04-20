#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../support/prefix_sum.h"

namespace {

constexpr int MinLog2Size = 2;
constexpr int MaxLog2Size = 28;
constexpr int MinValue = -10;
constexpr int MaxValue = 10;

void reference_prefix_sum(const int* input, int* output, int n) {
    if (n <= 0) {
        return;
    }

    output[0] = input[0];
    for (int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <log2_input_size>\n";
        return EXIT_FAILURE;
    }

    int log2_size = 0;
    try {
        log2_size = std::stoi(argv[1]);
    } catch (const std::exception&) {
        std::cerr << "Invalid log2 input size: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    if (log2_size < MinLog2Size || log2_size > MaxLog2Size) {
        std::cerr << "log2_input_size must be in range ["
                  << MinLog2Size << ", " << MaxLog2Size << "]\n";
        return EXIT_FAILURE;
    }

    if (log2_size >= std::numeric_limits<int>::digits) {
        std::cerr << "log2_input_size " << log2_size
                  << " is too large for the current int-based interface\n";
        return EXIT_FAILURE;
    }

    const int n = 1 << log2_size;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> value_dist(MinValue, MaxValue);

    std::vector<int> input(n);
    std::vector<int> output(n, 0);
    std::vector<int> reference(n, 0);

    for (int i = 0; i < n; ++i) {
        input[i] = value_dist(gen);
    }

    reference_prefix_sum(input.data(), reference.data(), n);

    const auto start = std::chrono::high_resolution_clock::now();
    prefix_sum(input.data(), output.data(), n);
    const auto end = std::chrono::high_resolution_clock::now();

    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (output[i] != reference[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": expected "
                      << reference[i] << ", got " << output[i] << '\n';
            break;
        }
    }

    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    std::cout << "Input log2 size: " << log2_size << '\n';
    std::cout << "Input size: " << n << '\n';
    std::cout << "Elapsed time (ms): " << elapsed_ms.count() << '\n';
    std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << '\n';

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
