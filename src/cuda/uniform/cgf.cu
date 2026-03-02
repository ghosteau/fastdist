// src/cuda/uniform/cgf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/uniform.cuh"
#include "fastdist/math/constants.h"

namespace fastdist::cuda::uniform {
    // Uniform MGF Device
    __device__ double uniform_mgf_device(const double t, const double a, const double b) {
        if (!isfinite(t) || !isfinite(a) || !isfinite(b) || a >= b) {
            return nan("");
        }

        if (t == 0.0) {
            return 1.0;
        }

        return (exp(b * t) - exp(a * t)) / (t * (b - a));
    }

    // CUDA kernel
    __global__ void uniform_cgf_kernel(const double* t, double* output, const int n, const double a, const double b, const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double t_val = t[idx] + stepSize * global_idx;

            if (!isfinite(t_val) || !isfinite(a) || !isfinite(b) || a >= b) {
                output[idx] = nan("");
                return;
            }

            const double mgf = uniform_mgf_device(t_val, a, b);
            if (mgf <= 0.0) {
                output[idx] = nan("");
                return;
            }

            output[idx] = log(mgf);
        }
    }

    // Dispatcher
    void uniform_cgf_dispatcher(const double* t, double* output, const int n, const double a, const double b, const double stepSize) {
        DeviceContext<double, double>& ctx = get_context<double, double>(n);

        execute_cuda_kernel<double, double>(
            uniform_cgf_kernel,
            t,
            output, ctx.dev_in, ctx.dev_out,
            n,
            StreamingThresholds::SIMPLE_MATH,
            a,
            b,
            stepSize);
    }
} // namespace fastdist::cuda::uniform
