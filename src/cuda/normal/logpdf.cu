// src/cuda/normal/logpdf.cu

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
    __global__ void normal_logpdf_kernel(const double* x, double* output, const int n, const double mu,
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

            const double inv_sigma = 1.0 / sigma;
            const double z = (x_val - mu) * inv_sigma;
            output[idx] = -0.5 * z * z - std::log(sigma) - LOG_SQRT_2PI;
        }
    }

    // Dispatcher
    void normal_logpdf_dispatcher(const double* x, double* output, const int n, const double mu, const double sigma,
                                  const double stepSize) {
        DeviceContext<double, double>& ctx = get_context<double, double>(n);
 execute_cuda_kernel<double, double>(normal_logpdf_kernel, x, output, ctx.dev_in, ctx.dev_out, n,
                                            StreamingThresholds::COMPLEX_MATH, mu, sigma, stepSize);
    }
} // namespace fastdist::cuda::normal
