// src/cuda/bernoulli/pmf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/executor.cuh"
#include "cuda/bernoulli.cuh"
#include "fastdist/math/constants.h"

namespace fastdist::cuda::bernoulli {
    // CUDA kernel
    __global__ void bernoulli_pmf_kernel(const int* k, double* output, const int n, const double p, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            int k_val = k[idx] + stepSize * global_idx;

            if (!isfinite(p) || p < 0.0 || p > 1.0) {
                output[idx] = nan("");
                return;
            }

            // Bernoulli is only defined for k = 0 or 1
            if (k_val != 0 && k_val != 1) {
                output[idx] = 0.0;
                return;
            }

            output[idx] = (k_val == 1) ? p : (1.0 - p);
        }
    }

    // Dispatcher
    void bernoulli_pmf_dispatcher(const int* k, double* output, const int n, const double p, const int stepSize) {
        DeviceContext<int, double>& ctx = get_context<int, double>(n);

        execute_cuda_kernel<int, double>(
            bernoulli_pmf_kernel,
            k,
            output, ctx.dev_in, ctx.dev_out,
            n,
            StreamingThresholds::SIMPLE_MATH,
            p,
            stepSize);
    }
} // namespace fastdist::cuda::bernoulli
