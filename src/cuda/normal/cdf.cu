// src/cuda/normal/cdf.cu

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
    __global__ void normal_cdf_kernel(const double* x, double* output, const int n, const double mu,
                                      const double sigma, const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double x_val = x[idx] + stepSize * static_cast<double>(global_idx);

            // Check for invalid inputs
            if (!isfinite(x_val) || !isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
                output[idx] = nan("");
                return;
            }

            const double z = (x_val - mu) / (sigma * std::sqrt(2.0));
            output[idx] = 0.5 * (1.0 + std::erf(z));
        }
    }

    // Dispatcher
    void normal_cdf_dispatcher(const double* x, double* output, const int n, const double mu, const double sigma, const double stepSize) {
        execute_cuda_kernel<double, double>(
            normal_cdf_kernel,
            x,
            output,
            n,
            StreamingThresholds::COMPLEX_MATH,
            mu,
            sigma,
            stepSize);
    }
} // namespace fastdist::cuda::normal
