// src/cuda/utils/euclidean_distance.cu

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/executor.cuh"
#include "fastdist/cuda/utils.cuh"

namespace fastdist::cuda::utils {

    // CUDA kernel: Logic for square root of summed squared differences
    __global__ void euclidean_distance_kernel(const double* x_input, const double* y_input, double* output,
                                              const int* strides, const int batch_count, const int offset) {
        const int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= batch_count) return;

        // Apply offset for streaming or partial batch logic
        const int actual_b = b + offset;
        const int start = strides[actual_b];
        const int end = strides[actual_b + 1];
        const int n = end - start;

        double sum_sq = 0.0;

        for (int i = 0; i < n; ++i) {
            // Indexing into flattened segments using start + i
            const double xv = x_input[start + i];
            const double yv = y_input[start + i];

            if (!isfinite(xv) || !isfinite(yv)) {
                output[actual_b] = nan("");
                return;
            }

            const double diff = xv - yv;
            sum_sq += diff * diff;
        }

        output[actual_b] = sqrt(sum_sq);
    }

    // Dispatcher: Manually inlined executor logic to bypass MSVC template issues
    void euclidean_distance_dispatcher(const double* x_input, const double* y_input, double* output, const int* strides,
                                       const int batch_count) {
        if (batch_count <= 0) return;

        double *d_x = nullptr, *d_y = nullptr, *d_output = nullptr;
        int* d_strides = nullptr;

        const int total_elements = strides[batch_count];
        const size_t inputSize = total_elements * sizeof(double);
        const size_t outputSize = batch_count * sizeof(double);
        const size_t stridesSize = (batch_count + 1) * sizeof(int);

        try {
            // Device Allocations
            if (cudaMalloc(&d_x, inputSize) != cudaSuccess) throw std::runtime_error("cudaMalloc d_x failed");
            if (cudaMalloc(&d_y, inputSize) != cudaSuccess) throw std::runtime_error("cudaMalloc d_y failed");
            if (cudaMalloc(&d_output, outputSize) != cudaSuccess)
                throw std::runtime_error("cudaMalloc d_output failed");
            if (cudaMalloc(&d_strides, stridesSize) != cudaSuccess)
                throw std::runtime_error("cudaMalloc d_strides failed");

            // Transfers to Device
            cudaMemcpy(d_x, x_input, inputSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y_input, inputSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_strides, strides, stridesSize, cudaMemcpyHostToDevice);

            // Kernel Config
            constexpr int threadsPerBlock = 256;
            const int blocksPerGrid = (batch_count + threadsPerBlock - 1) / threadsPerBlock;
            const int offset = 0;


            euclidean_distance_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_output, d_strides, batch_count,
                                                                          offset);

            // Verify Launch and Sync
            if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("Euclidean kernel launch failed");
            if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("Euclidean kernel execution failed");

            // Transfer Result back to Host
            cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

        } catch (...) {
            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_output);
            cudaFree(d_strides);
            throw;
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_output);
        cudaFree(d_strides);
    }
} // namespace fastdist::cuda::utils
