// src/cuda/normal/mgf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/normal.cuh"
#include "fastdist/math/constants.h"

namespace fastdist::cuda::normal {
    // CUDA kernel
    __global__ void normal_mgf_kernel(const double* t, double* output, const int n, const double mu, const double sigma,
                                      const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double t_val = t[idx] + stepSize * static_cast<double>(global_idx);

            // Check for invalid inputs
            if (!isfinite(t_val) || !isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
                output[idx] = nan("");
                return;
            }

            output[idx] = std::exp(mu * t_val + 0.5 * sigma * sigma * t_val * t_val);
        }
    }

    // Dispatcher
    void normal_mgf_dispatcher(const double* t, double* output, const int n, const double mu, const double sigma,
                               const double stepSize) {
        DeviceContext<double, double>& ctx = get_context<double, double>(n);
        execute_cuda_kernel<double, double>(normal_mgf_kernel, t, output, ctx.dev_in, ctx.dev_out, n,
                                            StreamingThresholds::COMPLEX_MATH, mu, sigma, stepSize);
    }
} // namespace fastdist::cuda::normal
