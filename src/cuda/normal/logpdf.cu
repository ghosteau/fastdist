// src/cuda/normal/logpdf.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "fastdist/cuda/normal.cuh"
#include <stdexcept>
#include <string>
#include "fastdist/math/constants.h"

namespace fastdist::cuda::normal {
    // CUDA kernel
    __global__ void normal_logpdf_kernel(const double* x, double* output, const int n, const double mu,
                                      const double sigma, const double stepSize, const int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int global_idx = idx + offset;

        if (idx < n) {
            double x_val = x[idx] + stepSize * static_cast<double>(global_idx);

            // Check for invalid inputs
            if (!isfinite(x_val) || !isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
                output[idx] = nan("");
                return;
            }

            const double inv_sigma = 1.0 / sigma;
            const double z = (x_val - mu) * inv_sigma;
            output[idx] = -0.5 * z * z - std::log(sigma) - LOG_SQRT_2PI;
        }
    }

    static void normal_logpdf_cuda_simple(const double* x, double* output, const int n, const double mu,
                                       const double sigma, const double stepSize) {
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
        normal_logpdf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_output, n, mu, sigma, stepSize, offset);

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

    static void normal_logpdf_cuda_streaming(const double* x, double* output, const int n, const double mu,
                                          const double sigma, const double stepSize) {
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

            normal_logpdf_kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_x + offset, d_output + offset,
                                                                          currentChunkSize, mu, sigma, stepSize, offset);

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
    void normal_logpdf_dispatcher(const double* x, double* output, const int n, const double mu, const double sigma, const double stepSize) {
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
            normal_logpdf_cuda_simple(x, output, n, mu, sigma, stepSize);
        } else {
            normal_logpdf_cuda_streaming(x, output, n, mu, sigma, stepSize);
        }
    }
} // namespace fastdist::cuda::normal
