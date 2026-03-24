// src/cuda/uniform/pdf.cu

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
    __global__ void uniform_pdf_kernel(const double* x, double* output, const int n, const double a, const double b, const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double x_val = x[idx] + stepSize * global_idx;

            if (!isfinite(x_val) || !isfinite(a) || !isfinite(b) || a >= b) {
                output[idx] = nan("");
                return;
            }

            // PDF is zero outside [a,b]
            if (x_val < a || x_val > b) {
                output[idx] = 0.0;
                return;
            }

            output[idx] = 1.0 / (b - a);
        }
    }

    // Dispatcher
    void uniform_pdf_dispatcher(const double* x, double* output, const int n, const double a, const double b, const double stepSize) {
        DeviceContext<double, double>& ctx = get_context<double, double>(n);

        execute_cuda_kernel<double, double>(
            uniform_pdf_kernel,
            x,
            output, ctx.dev_in, ctx.dev_out,
            n,
            StreamingThresholds::SIMPLE_MATH,
            a,
            b,
            stepSize);
    }
} // namespace fastdist::cuda::uniform
