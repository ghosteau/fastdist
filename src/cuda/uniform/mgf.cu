// src/cuda/uniform/mgf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/uniform.cuh"
#include "fastdist/math/constants.h"

namespace fastdist::cuda::uniform {
    // CUDA kernel
    __global__ void uniform_mgf_kernel(const double* t, double* output, const int n, const double a, const double b,
                                       const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double t_val = t[idx] + stepSize * global_idx;

            if (!isfinite(t_val) || !isfinite(a) || !isfinite(b) || a >= b) {
                output[idx] = nan("");
                return;
            }

            if (t_val == 0.0) {
                output[idx] = 1.0;
                return;
            }

            output[idx] = (exp(b * t_val) - exp(a * t_val)) / (t_val * (b - a));
        }
    }

    // Dispatcher
    void uniform_mgf_dispatcher(const double* t, double* output, const int n, const double a, const double b,
                                const double stepSize) {
        DeviceContext<double, double>& ctx = get_context<double, double>(n);

        execute_cuda_kernel<double, double>(uniform_mgf_kernel, t, output, ctx.dev_in, ctx.dev_out, n,
                                            StreamingThresholds::COMPLEX_MATH, a, b, stepSize);
    }
} // namespace fastdist::cuda::uniform
