// src/cuda/poisson/cdf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "fastdist/cuda/poisson.cuh"
#include "cuda/executor.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::poisson {
    // Poisson PMF Device
    __device__ double poisson_pmf_device(const int k, const double lambda) {
        if (lambda <= 0.0 || k < 0) {
            return 0.0;
        }

        // Compute log PMF for numerical stability
        double log_p = k * log(lambda) - lambda - lgamma(static_cast<double>(k + 1));
        return exp(log_p);
    }

    // CUDA kernel
    __global__ void poisson_cdf_kernel(const double* x, double* output, const int n, const double lambda, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double x_val = x[idx] + stepSize * global_idx;

            if (!isfinite(x_val) || !isfinite(lambda) || lambda <= 0.0) {
                output[idx] = nan("");
                return;
            }

            if (x_val < 0.0) {
                output[idx] = 0.0;
                return;
            }

            const int ki = static_cast<int>(std::floor(x_val));

            double sum = 0.0;
            for (int i = 0; i <= ki; ++i) {
                sum += poisson_pmf_device(i, lambda);
            }

            output[idx] = sum;
        }
    }

    // Dispatcher
    void poisson_cdf_dispatcher(const double* x, double* output, const int n, const double lambda, const int stepSize) {
        execute_cuda_kernel<double, double>(
            poisson_cdf_kernel,
            x,
            output,
            n,
            StreamingThresholds::SIMPLE_MATH,
            lambda,
            stepSize);
    }
} // namespace fastdist::cuda::poisson
