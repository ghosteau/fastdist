// src/cuda/poisson/pmf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/poisson.cuh"
#include "fastdist/math/constants.h"

namespace fastdist::cuda::poisson {
    // CUDA kernel
    __global__ void poisson_pmf_kernel(const double* x, double* output, const int n, const double lambda, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double x_val = x[idx] + stepSize * global_idx;

            if (!isfinite(x_val) || !isfinite(lambda) || lambda <= 0.0) {
                output[idx] = nan("");
                return;
            }

            if (x_val < 0.0 || floor(x_val) != x_val) {
                output[idx] = 0.0;
                return;
            }

            const double log_p = x_val * log(lambda) - lambda - lgamma(x_val + 1);
            output[idx] = exp(log_p);
        }
    }

    // Dispatcher
    void poisson_pmf_dispatcher(const double* x, double* output, const int n, const double lambda, const int stepSize) {
        execute_cuda_kernel<double, double>(
            poisson_pmf_kernel,
            x,
            output,
            n,
            StreamingThresholds::COMPLEX_MATH,
            lambda,
            stepSize);
    }
} // namespace fastdist::cuda::poisson
