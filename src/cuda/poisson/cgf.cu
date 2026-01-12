// src/cuda/poisson/cgf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "fastdist/cuda/poisson.cuh"
#include "cuda/executor.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::poisson {
    // CUDA kernel
    __global__ void poisson_cgf_kernel(const double* t, double* output, const int n, const double lambda, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double t_val = t[idx] + stepSize * global_idx;

            if (!isfinite(t_val) || !isfinite(lambda) || lambda <= 0.0) {
                output[idx] = nan("");
                return;
            }

            output[idx] = lambda * (exp(t_val) - 1.0);
        }
    }

    // Dispatcher
    void poisson_cgf_dispatcher(const double* t, double* output, const int n, const double lambda, const int stepSize) {
        execute_cuda_kernel<double, double>(
            poisson_cgf_kernel,
            x,
            output,
            n,
            StreamingThresholds::SIMPLE_MATH,
            lambda,
            stepSize);
    }
} // namespace fastdist::cuda::poisson
