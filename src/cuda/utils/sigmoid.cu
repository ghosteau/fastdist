// src/cuda/utils/sigmoid.cu

#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/utils.cuh"

namespace fastdist::cuda::utils {
    // CUDA kernel
    __global__ void sigmoid_kernel(const double* x, double* output, const int n, const int offset) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            const double x_val = x[idx];
            if (!isfinite(x_val)) {
                output[idx] = nan("");
                return;
            }

            // Numerically stable sigmoid
            if (x_val >= 0.0) {
                const double z = exp(-x_val);
                output[idx] = 1.0 / (1.0 + z);
                return;
            }

            const double z = exp(x_val);
            output[idx] = z / (1.0 + z);
        }
    }

    // Dispatcher
    void sigmoid_dispatcher(const double* x, double* output, const int n) {
        DeviceContext<double, double>& ctx = get_context<double, double>(n);

        execute_cuda_kernel<double, double>(sigmoid_kernel, x, output, ctx.dev_in, ctx.dev_out, n,
                                            StreamingThresholds::COMPLEX_MATH);
    }
} // namespace fastdist::cuda::utils
