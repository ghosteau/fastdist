// src/cuda/bernoulli/cgf.cu

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
    __global__ void bernoulli_cgf_kernel(const double* t, double* output, const int n, const double p, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double t_val = t[idx] + stepSize * global_idx;

            if (!isfinite(t_val) || !isfinite(p) || p < 0.0 || p > 1.0) {
                output[idx] = nan("");
                return;
            }

            output[idx] = std::log((1.0 - p) + p * std::exp(t_val));
        }
    }

    // Dispatcher
    void bernoulli_cgf_dispatcher(const double* t, double* output, const int n, const double p, const int stepSize) {
        execute_cuda_kernel<double, double>(
            bernoulli_cgf_kernel,
            t,
            output,
            n,
            StreamingThresholds::COMPLEX_MATH,
            p,
            stepSize);
    }
} // namespace fastdist::cuda::bernoulli
