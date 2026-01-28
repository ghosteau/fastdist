// src/cuda/exponential/mgf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/exponential.cuh"
#include "fastdist/math/constants.h"

namespace fastdist::cuda::exponential {
    // CUDA kernel
    __global__ void exponential_mgf_kernel(const double* t, double* output, const int n, const double lambda, const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double t_val = t[idx] + stepSize * global_idx;

            if (!isfinite(t_val) || !isfinite(lambda) || lambda <= 0.0 || t_val >= lambda) {
                output[idx] = nan("");
                return;
            }

            output[idx] = lambda / (lambda - t_val);
        }
    }

    // Dispatcher
    void exponential_mgf_dispatcher(const double* t, double* output, const int n, const double lambda, const double stepSize) {
        execute_cuda_kernel<double, double>(
            exponential_mgf_kernel,
            t,
            output,
            n,
            StreamingThresholds::SIMPLE_MATH,
            lambda,
            stepSize);
    }
} // namespace fastdist::cuda::exponential
