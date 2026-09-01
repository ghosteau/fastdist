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
        DeviceContext<double, double>& ctx = get_context<double, double>(n);

        execute_cuda_kernel<double, double>(logit_kernel, p, output, ctx.dev_in, ctx.dev_out, n,
                                            StreamingThresholds::COMPLEX_MATH);
    }
} // namespace fastdist::cuda::utils
