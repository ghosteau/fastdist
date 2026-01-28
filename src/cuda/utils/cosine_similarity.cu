// src/cuda/utils/cosine_similarity.cu

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/utils.cuh"

namespace fastdist::cuda::utils {

    // CUDA kernel remains essentially the same
    __global__ void cosine_similarity_kernel(const double* x_input, const double* y_input, double* output,
                                             const int* strides, const int batch_count, const int offset) {
        const int b = blockIdx.x * blockDim.x + threadIdx.x;
        // The b index here represents the batch index relative to the current launch
        if (b >= batch_count) return;

        // Apply offset to batch index to find correct strides
        const int actual_b = b + offset;
        const int start = strides[actual_b];
        const int end = strides[actual_b + 1];

        // n is the number of elements in the specific vectors for this batch
        const int n = end - start;

        double dot = 0.0;
        double norm_x = 0.0;
        double norm_y = 0.0;

        for (int i = 0; i < n; ++i) {
            const double xv = x_input[start + i];
            const double yv = y_input[start + i];

            if (!isfinite(xv) || !isfinite(yv)) {
                output[actual_b] = nan("");
                return;
            }

            dot += xv * yv;
            norm_x += xv * xv;
            norm_y += yv * yv;
        }

        if (norm_x == 0.0 || norm_y == 0.0) {
            output[actual_b] = nan("");
            return;
        }

        output[actual_b] = dot / (sqrt(norm_x) * sqrt(norm_y));
    }

    // Consolidated Dispatcher: Replaces the template call with localized logic
    void cosine_similarity_dispatcher(const double* x_input, const double* y_input, double* output, const int* strides,
                                      const int batch_count) {
        if (batch_count <= 0) return;

        double *d_x = nullptr, *d_y = nullptr, *d_output = nullptr;
        int* d_strides = nullptr;

        // total_elements is stored at the end of the strides array
        const int total_elements = strides[batch_count];
        const size_t inputSize = total_elements * sizeof(double);
        const size_t outputSize = batch_count * sizeof(double);
        const size_t stridesSize = (batch_count + 1) * sizeof(int);

        try {
            // 1. Allocation
            if (cudaMalloc(&d_x, inputSize) != cudaSuccess) throw std::runtime_error("cudaMalloc d_x failed");
            if (cudaMalloc(&d_y, inputSize) != cudaSuccess) throw std::runtime_error("cudaMalloc d_y failed");
            if (cudaMalloc(&d_output, outputSize) != cudaSuccess)
                throw std::runtime_error("cudaMalloc d_output failed");
            if (cudaMalloc(&d_strides, stridesSize) != cudaSuccess)
                throw std::runtime_error("cudaMalloc d_strides failed");

            // 2. Host to Device Transfer
            cudaMemcpy(d_x, x_input, inputSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y_input, inputSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_strides, strides, stridesSize, cudaMemcpyHostToDevice);

            // 3. Kernel Launch Parameters
            // Note: We are parallelizing over the batch_count (one thread per similarity calculation)
            constexpr int threadsPerBlock = 256;
            const int blocksPerGrid = (batch_count + threadsPerBlock - 1) / threadsPerBlock;
            const int offset = 0;

            // This is the line MSVC hated, but here it's inside a .cu file, so it's safe!
            cosine_similarity_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_output, d_strides, batch_count,
                                                                         offset);

            // 4. Error Checking & Sync
            if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("Kernel launch failed");
            if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("Kernel execution failed");

            // 5. Device to Host Transfer
            cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

        } catch (...) {
            // Cleanup on any error
            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_output);
            cudaFree(d_strides);
            throw;
        }

        // Normal Cleanup
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_output);
        cudaFree(d_strides);
    }
} // namespace fastdist::cuda::utils
