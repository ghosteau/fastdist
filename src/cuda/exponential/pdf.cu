// src/cuda/exponential/pdf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda/exponential.cuh"
#include "cuda/executor.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::exponential {
     // CUDA kernel
    __global__ void exponential_pdf_kernel(const double* x, double* output, const int n, const double lambda, const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double x_val = x[idx] + stepSize * global_idx;

            if (!isfinite(x_val) || !isfinite(lambda) || lambda <= 0.0) {
                output[idx] = nan("");
                return;
            }

            if (x_val < 0) {
                output[idx] = 0.0;
                return;
            }

            output[idx] = lambda * exp(-lambda * x_val);
        }
    }

    // Dispatcher
    void exponential_pdf_dispatcher(const double* x, double* output, const int n, const double lambda, const double stepSize) {
        execute_cuda_kernel<double, double>(
            exponential_pdf_kernel,
            x,
            output,
            n,
            StreamingThresholds::COMPLEX_MATH,
            lambda,
            stepSize);
    }
} // namespace fastdist::cuda::exponential
