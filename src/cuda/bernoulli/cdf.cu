// src/cuda/bernoulli/cdf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "fastdist/cuda/bernoulli.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::bernoulli {
    constexpr size_t INPUT_SIZE = sizeof(int);
    constexpr size_t OUTPUT_SIZE = sizeof(double);
    
    // CUDA kernel
    __global__ void bernoulli_cdf_kernel(const int* k, double* output, const int n, const double p, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            int k_val = k[idx] + stepSize * global_idx;

            if (!isfinite(p) || p < 0.0 || p > 1.0) {
                output[idx] = nan("");
                return;
            }

            if (k_val < 0) {
                output[idx] = 0.0;
                return;
            }
            if (k_val < 1) {
                output[idx] = 1.0 - p;
                return;
            }

            output[idx] = 1.0;
        }
    }

    static void bernoulli_cdf_cuda_simple(const int* k, double* output, const int n, const double p, const int stepSize) {
        int *d_k;
        double *d_output;

        const size_t totalInputSize = n * INPUT_SIZE;
        const size_t totalOutputSize = n * OUTPUT_SIZE;

        cudaError_t err = cudaMalloc(&d_k, totalInputSize);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc d_k failed: ") + cudaGetErrorString(err));
        }

        err = cudaMalloc(&d_output, totalOutputSize);
        if (err != cudaSuccess) {
            cudaFree(d_k);
            throw std::runtime_error(std::string("cudaMalloc d_output failed: ") + cudaGetErrorString(err));
        }

        // Sending info to GPU
        err = cudaMemcpy(d_k, k, totalInputSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_k);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaMemcpy Host->Device failed: ") + cudaGetErrorString(err));
        }

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        const int offset = 0;
        bernoulli_cdf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_k, d_output, n, p, stepSize, offset);

        // Check for Kernel Errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_k);
            cudaFree(d_output);
            throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
        }

        // Ensure kernel execution is complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_k);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
        }

        // Receiving info from GPU
        err = cudaMemcpy(output, d_output, totalOutputSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_k);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaMemcpy Device->Host failed: ") + cudaGetErrorString(err));
        }

        cudaFree(d_k);
        cudaFree(d_output);
    }

    static void bernoulli_cdf_cuda_streaming(const int* k, double* output, const int n, const double p, const int stepSize) {
        static const int STREAM_COUNT = 4;
        int *d_k = nullptr;
        double *d_output = nullptr;

        const size_t totalInputSize = n * INPUT_SIZE;
        const size_t totalOutputSize = n * OUTPUT_SIZE;

        // Create CUDA streams
        cudaStream_t streams[STREAM_COUNT];
        for (int i = 0; i < STREAM_COUNT; i++) {
            cudaError_t err = cudaStreamCreate(&streams[i]);
            if (err != cudaSuccess) {
                // Cleanup already created streams
                for (int j = 0; j < i; j++) {
                    cudaStreamDestroy(streams[j]);
                }
                throw std::runtime_error(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err));
            }
        }

        const int chunkSize = (n + STREAM_COUNT - 1) / STREAM_COUNT;
        const int threadsPerBlock = 256;

        cudaError_t err = cudaMalloc(&d_k, totalInputSize);
        if (err != cudaSuccess) {
            for (int i = 0; i < STREAM_COUNT; i++) {
                cudaStreamDestroy(streams[i]);
            }
            throw std::runtime_error(std::string("cudaMalloc d_k failed: ") + cudaGetErrorString(err));
        }

        err = cudaMalloc(&d_output, totalOutputSize);
        if (err != cudaSuccess) {
            cudaFree(d_k);
            for (int i = 0; i < STREAM_COUNT; i++) {
                cudaStreamDestroy(streams[i]);
            }
            throw std::runtime_error(std::string("cudaMalloc d_output failed: ") + cudaGetErrorString(err));
        }

        for (int i = 0; i < STREAM_COUNT; i++) {
            const int offset = i * chunkSize;
            const int currentChunkSize = std::min(chunkSize, n - offset);

            if (currentChunkSize <= 0) break;

            const size_t chunkInputBytes = currentChunkSize * INPUT_SIZE;
            const size_t chunkOutputBytes = currentChunkSize * OUTPUT_SIZE;
            const int blocks = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;

            // Sending info to GPU
            cudaMemcpyAsync(d_k + offset, k + offset, chunkInputBytes, cudaMemcpyHostToDevice, streams[i]);

            bernoulli_cdf_kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_k + offset, d_output + offset,
                                                                          currentChunkSize, p, stepSize, offset);

            // Receiving info from GPU
            cudaMemcpyAsync(output + offset, d_output + offset, chunkOutputBytes, cudaMemcpyDeviceToHost, streams[i]);
        }

        for (int i = 0; i < STREAM_COUNT; i++) {
            err = cudaStreamSynchronize(streams[i]);
            if (err != cudaSuccess) {
                // Cleanup everything
                cudaFree(d_k);
                cudaFree(d_output);
                for (int j = 0; j < STREAM_COUNT; j++) {
                    cudaStreamDestroy(streams[j]);
                }
                throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err));
            }
            cudaStreamDestroy(streams[i]);
        }

        cudaFree(d_k);
        cudaFree(d_output);
    }

    // Dispatcher
    void bernoulli_cdf_dispatcher(const int* k, double* output, const int n, const double p, const int stepSize) {
        if (n <= 0) return;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA has sticky error: %s\n", cudaGetErrorString(err));
            cudaDeviceReset();
        }

        // Threshold for using streaming version
        // Tune this based on your GPU and typical workload
        static const int STREAMING_THRESHOLD = 100000;

        if (n < STREAMING_THRESHOLD) {
            bernoulli_cdf_cuda_simple(k, output, n, p, stepSize);
        } else {
            bernoulli_cdf_cuda_streaming(k, output, n, p, stepSize);
        }
    }
} // namespace fastdist::cuda::bernoulli
