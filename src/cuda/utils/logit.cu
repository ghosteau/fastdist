// src/cuda/utils/logit.cu

#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/utils.cuh"

namespace fastdist::cuda::utils {
    // CUDA kernel
    __global__ void logit_kernel(const double* p, double* output, const int n, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double p_val = p[idx];
            if (!isfinite(p_val) || p_val <= 0.0 || p_val >= 1.0) {
                output[idx] = nan("");
                return;
            }

            output[idx] = log(p_val / (1.0 - p_val));
        }
    }

    // Dispatcher
    void logit_dispatcher(const double* p, double* output, const int n) {
        execute_cuda_kernel<double, double>(logit_kernel, p, output, n, StreamingThresholds::COMPLEX_MATH);
    }
} // namespace fastdist::cuda::utils
