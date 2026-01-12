// src/cuda/bernoulli/cdf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "fastdist/cuda/bernoulli.cuh"
#include "cuda/executor.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::bernoulli {
    // CUDA kernel
    __global__ void bernoulli_cdf_kernel(const int* k, double* output, const int n, const double p, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            int k_val = k[idx] + stepSize * global_idx;

            if (!isfinite(p) || p < 0.0 || p > 1.0) {
                output[idx] = nan("");
                return;
            }

            if (k_val < 0) {
                output[idx] = 0.0;
                return;
            }
            if (k_val < 1) {
                output[idx] = 1.0 - p;
                return;
            }

            output[idx] = 1.0;
        }
    }

    // Dispatcher
    void bernoulli_cdf_dispatcher(const int* k, double* output, const int n, const double p, const int stepSize) {
        execute_cuda_kernel<int, double>(
            bernoulli_cdf_kernel,
            k,
            output,
            n,
            StreamingThresholds::SIMPLE_MATH,
            p,
            stepSize);
    }
} // namespace fastdist::cuda::bernoulli
