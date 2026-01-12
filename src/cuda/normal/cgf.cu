// src/cuda/normal/cgf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "fastdist/cuda/normal.cuh"
#include "cuda/executor.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::normal {
    // CUDA kernel
    __global__ void normal_cgf_kernel(const double* t, double* output, const int n, const double mu,
                                      const double sigma, const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double t_val = t[idx] + stepSize * static_cast<double>(global_idx);

            // Check for invalid inputs
            if (!isfinite(t_val) || !isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
                output[idx] = nan("");
                return;
            }
            
            output[idx] = mu * t_val + 0.5 * sigma * sigma * t_val * t_val;
        }
    }

    // Dispatcher
    void normal_cgf_dispatcher(const double* t, double* output, const int n, const double mu, const double sigma, const double stepSize) {
        execute_cuda_kernel<double, double>(
            normal_cgf_kernel,
            t,
            output,
            n,
            StreamingThresholds::SIMPLE_MATH,
            mu,
            sigma,
            stepSize);
    }
} // namespace fastdist::cuda::normal
