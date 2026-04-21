#include <algorithm>
#include <cstring>
#include <vector>

#include <omp.h>

namespace multicore_prefix_sum {

namespace {

inline int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void copy_input_and_zero_pad(const int* input, int n, std::vector<int>& buffer) {
    std::fill(buffer.begin(), buffer.end(), 0);
    std::memcpy(buffer.data(), input, static_cast<std::size_t>(n) * sizeof(int));
}

void exclusive_to_inclusive(const int* input,
                            const std::vector<int>& exclusive,
                            int* output,
                            int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        output[i] = exclusive[i] + input[i];
    }
}

}  // namespace

void prefix_sum_kogge_stone(const int* input, int* output, int n) {
    if (n <= 0) return;

    std::vector<int> curr(input, input + n);
    std::vector<int> next(n);

    for (int offset = 1; offset < n; offset <<= 1) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (i >= offset) {
                next[i] = curr[i] + curr[i - offset];
            } else {
                next[i] = curr[i];
            }
        }
        curr.swap(next);
    }

    std::memcpy(output, curr.data(), static_cast<std::size_t>(n) * sizeof(int));
}

void prefix_sum_blelloch(const int* input, int* output, int n) {
    if (n <= 0) return;

    const int m = next_power_of_two(n);
    std::vector<int> tree(m);
    copy_input_and_zero_pad(input, n, tree);

    for (int stride = 1; stride < m; stride <<= 1) {
        const int step = stride << 1;
#pragma omp parallel for schedule(static)
        for (int k = 0; k < m; k += step) {
            const int right = k + step - 1;
            const int left  = k + stride - 1;
            tree[right] += tree[left];
        }
    }

    tree[m - 1] = 0;

    for (int stride = m >> 1; stride >= 1; stride >>= 1) {
        const int step = stride << 1;
#pragma omp parallel for schedule(static)
        for (int k = 0; k < m; k += step) {
            const int right = k + step - 1;
            const int left  = k + stride - 1;
            const int t = tree[left];
            tree[left] = tree[right];
            tree[right] += t;
        }
    }

    exclusive_to_inclusive(input, tree, output, n);
}

void prefix_sum_brent_kung(const int* input, int* output, int n) {
    if (n <= 0) return;

    const int m = next_power_of_two(n);
    std::vector<int> tree(m);
    copy_input_and_zero_pad(input, n, tree);

    for (int stride = 1; stride < m; stride <<= 1) {
        const int step = stride << 1;
#pragma omp parallel for schedule(static)
        for (int i = step - 1; i < m; i += step) {
            tree[i] += tree[i - stride];
        }
    }

    tree[m - 1] = 0;

    for (int stride = m >> 1; stride >= 1; stride >>= 1) {
        const int step = stride << 1;
#pragma omp parallel for schedule(static)
        for (int i = step - 1; i < m; i += step) {
            const int left = i - stride;
            const int t = tree[left];
            tree[left] = tree[i];
            tree[i] += t;
        }
    }

    exclusive_to_inclusive(input, tree, output, n);
}

}  // namespace multicore_prefix_sum