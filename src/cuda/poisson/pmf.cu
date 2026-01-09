// src/cuda/cuda/pmf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "fastdist/cuda/poisson.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::poisson {
    // CUDA kernel
    __global__ void poisson_pmf_kernel(const double* x, double* output, const int n, const double lambda, const int stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double x_val = x[idx] + stepSize * global_idx;

            if (!isfinite(x_val) || !isfinite(lambda) || lambda <= 0.0) {
                output[idx] = nan("");
                return;
            }

            if (x_val < 0.0 || floor(x_val) != x_val) {
                output[idx] = 0.0;
                return;
            }

            const double log_p = x_val * log(lambda) - lambda - lgamma(x_val + 1);
            output[idx] = exp(log_p);
        }
    }

    static void poisson_pmf_cuda_simple(const double* x, double* output, const int n, const double lambda, const int stepSize) {
        double *d_x, *d_output;
        const size_t totalSize = n * sizeof(double);

        cudaError_t err = cudaMalloc(&d_x, totalSize);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc d_x failed: ") + cudaGetErrorString(err));
        }

        err = cudaMalloc(&d_output, totalSize);
        if (err != cudaSuccess) {
            cudaFree(d_x);
            throw std::runtime_error(std::string("cudaMalloc d_output failed: ") + cudaGetErrorString(err));
        }

        // Sending info to GPU
        err = cudaMemcpy(d_x, x, totalSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_x);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaMemcpy Host->Device failed: ") + cudaGetErrorString(err));
        }

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        const int offset = 0;
        poisson_pmf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_output, n, lambda, stepSize, offset);

        // Check for Kernel Errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_x);
            cudaFree(d_output);
            throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
        }

        // Ensure kernel execution is complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_x);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
        }

        // Receiving info from GPU
        err = cudaMemcpy(output, d_output, totalSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_x);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaMemcpy Device->Host failed: ") + cudaGetErrorString(err));
        }

        cudaFree(d_x);
        cudaFree(d_output);
    }

    static void poisson_pmf_cuda_streaming(const double* x, double* output, const int n, const double lambda, const int stepSize) {
        static const int STREAM_COUNT = 4;
        double *d_x = nullptr, *d_output = nullptr;

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

        const size_t totalSize = n * sizeof(double);

        cudaError_t err = cudaMalloc(&d_x, totalSize);
        if (err != cudaSuccess) {
            for (int i = 0; i < STREAM_COUNT; i++) {
                cudaStreamDestroy(streams[i]);
            }
            throw std::runtime_error(std::string("cudaMalloc d_x failed: ") + cudaGetErrorString(err));
        }

        err = cudaMalloc(&d_output, totalSize);
        if (err != cudaSuccess) {
            cudaFree(d_x);
            for (int i = 0; i < STREAM_COUNT; i++) {
                cudaStreamDestroy(streams[i]);
            }
            throw std::runtime_error(std::string("cudaMalloc d_output failed: ") + cudaGetErrorString(err));
        }

        for (int i = 0; i < STREAM_COUNT; i++) {
            const int offset = i * chunkSize;
            const int currentChunkSize = std::min(chunkSize, n - offset);

            if (currentChunkSize <= 0) break;

            const size_t chunkBytes = currentChunkSize * sizeof(double);
            const int blocks = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;

            // Sending info to GPU
            cudaMemcpyAsync(d_x + offset, x + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]);

            poisson_pmf_kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_x + offset, d_output + offset,
                                                                          currentChunkSize, lambda, stepSize, offset);

            // Receiving info from GPU
            cudaMemcpyAsync(output + offset, d_output + offset, chunkBytes, cudaMemcpyDeviceToHost, streams[i]);
        }

        for (int i = 0; i < STREAM_COUNT; i++) {
            err = cudaStreamSynchronize(streams[i]);
            if (err != cudaSuccess) {
                // Cleanup everything
                cudaFree(d_x);
                cudaFree(d_output);
                for (int j = 0; j < STREAM_COUNT; j++) {
                    cudaStreamDestroy(streams[j]);
                }
                throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err));
            }
            cudaStreamDestroy(streams[i]);
        }

        cudaFree(d_x);
        cudaFree(d_output);
    }

    // Dispatcher
    void poisson_pmf_dispatcher(const double* x, double* output, const int n, const double lambda, const int stepSize) {
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
            poisson_pmf_cuda_simple(x, output, n, lambda, stepSize);
        } else {
            poisson_pmf_cuda_streaming(x, output, n, lambda, stepSize);
        }
    }
} // namespace fastdist::cuda::poisson
