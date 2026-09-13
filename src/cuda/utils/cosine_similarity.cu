// src/cuda/utils/cosine_similarity.cu

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "fastdist/cuda/utils.cuh"

namespace fastdist::cuda::utils {

    // One thread per vector pair. strides[b]..strides[b + 1] delimits pair b in
    // the flattened inputs; offset shifts b when a launch covers a sub-range.
    __global__ void cosine_similarity_kernel(const double* x_input, const double* y_input, double* output,
                                             const int* strides, const int batch_count, const int offset) {
        const int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= batch_count) return;

        const int actual_b = b + offset;
        const int start = strides[actual_b];
        const int end = strides[actual_b + 1];

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

    // Host entry point: copies the batch to the device, launches one thread per
    // pair, and copies the results back. Device buffers are freed on every path.
    void cosine_similarity_dispatcher(const double* x_input, const double* y_input, double* output, const int* strides,
                                      const int batch_count) {
        if (batch_count <= 0) return;

        double *d_x = nullptr, *d_y = nullptr, *d_output = nullptr;
        int* d_strides = nullptr;

        // strides has batch_count + 1 entries; the last is the total element count.
        const int total_elements = strides[batch_count];
        const size_t inputSize = total_elements * sizeof(double);
        const size_t outputSize = batch_count * sizeof(double);
        const size_t stridesSize = (batch_count + 1) * sizeof(int);

        try {
            if (cudaMalloc(&d_x, inputSize) != cudaSuccess) throw std::runtime_error("cudaMalloc d_x failed");
            if (cudaMalloc(&d_y, inputSize) != cudaSuccess) throw std::runtime_error("cudaMalloc d_y failed");
            if (cudaMalloc(&d_output, outputSize) != cudaSuccess)
                throw std::runtime_error("cudaMalloc d_output failed");
            if (cudaMalloc(&d_strides, stridesSize) != cudaSuccess)
                throw std::runtime_error("cudaMalloc d_strides failed");

            cudaMemcpy(d_x, x_input, inputSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y_input, inputSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_strides, strides, stridesSize, cudaMemcpyHostToDevice);

            constexpr int threadsPerBlock = 256;
            const int blocksPerGrid = (batch_count + threadsPerBlock - 1) / threadsPerBlock;
            const int offset = 0;

            cosine_similarity_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_output, d_strides, batch_count,
                                                                         offset);

            if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("Kernel launch failed");
            if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("Kernel execution failed");

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
