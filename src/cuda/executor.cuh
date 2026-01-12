// src/cuda/common/executor.cuh
#ifndef FASTDIST_EXECUTOR_CUH
#define FASTDIST_EXECUTOR_CUH

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fastdist::cuda {
    // Default streaming thresholds for different operation types
    struct StreamingThresholds {
        static constexpr int SIMPLE_MATH = 200000; // exp, log, sqrt, etc.
        static constexpr int COMPLEX_MATH = 100000; // trigonometric, special functions
        static constexpr int VERY_COMPLEX = 50000; // iterative algorithms, numerical integration
        static constexpr int MEMORY_BOUND = 150000; // operations limited by memory bandwidth
    };

    template<typename InputT, typename OutputT, typename KernelFunc, typename... Args>
    static void execute_simple(KernelFunc kernel, const InputT* input, OutputT* output, const int n, Args... args) {
        InputT* d_input = nullptr;
        OutputT* d_output = nullptr;
        const size_t inputSize = n * sizeof(InputT);
        const size_t outputSize = n * sizeof(OutputT);

        // Allocate device input memory
        cudaError_t err = cudaMalloc(&d_input, inputSize);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc d_input failed: ") + cudaGetErrorString(err));
        }

        // Allocate device output memory
        err = cudaMalloc(&d_output, outputSize);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            throw std::runtime_error(std::string("cudaMalloc d_output failed: ") + cudaGetErrorString(err));
        }

        // Copy input to device
        err = cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaMemcpy Host->Device failed: ") + cudaGetErrorString(err));
        }

        // Launch kernel
        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        const int offset = 0;

        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, args..., offset);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
        }

        // Wait for kernel completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
        }

        // Copy output back to host
        err = cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error(std::string("cudaMemcpy Device->Host failed: ") + cudaGetErrorString(err));
        }

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);
    }

    template<typename InputT, typename OutputT, typename KernelFunc, typename... Args>
    static void execute_streaming(KernelFunc kernel, const InputT* input, OutputT* output, const int n, Args... args) {
        static const int STREAM_COUNT = 4;
        InputT* d_input = nullptr;
        OutputT* d_output = nullptr;

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
        const size_t inputSize = n * sizeof(InputT);
        const size_t outputSize = n * sizeof(OutputT);

        // Allocate device input memory
        cudaError_t err = cudaMalloc(&d_input, inputSize);
        if (err != cudaSuccess) {
            for (int i = 0; i < STREAM_COUNT; i++) {
                cudaStreamDestroy(streams[i]);
            }
            throw std::runtime_error(std::string("cudaMalloc d_input failed: ") + cudaGetErrorString(err));
        }

        // Allocate device output memory
        err = cudaMalloc(&d_output, outputSize);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            for (int i = 0; i < STREAM_COUNT; i++) {
                cudaStreamDestroy(streams[i]);
            }
            throw std::runtime_error(std::string("cudaMalloc d_output failed: ") + cudaGetErrorString(err));
        }

        // Process chunks asynchronously
        for (int i = 0; i < STREAM_COUNT; i++) {
            const int offset = i * chunkSize;
            const int currentChunkSize = std::min(chunkSize, n - offset);

            if (currentChunkSize <= 0) break;

            const size_t inputChunkBytes = currentChunkSize * sizeof(InputT);
            const size_t outputChunkBytes = currentChunkSize * sizeof(OutputT);
            const int blocks = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;

            // Async copy to device
            cudaMemcpyAsync(d_input + offset, input + offset, inputChunkBytes, cudaMemcpyHostToDevice, streams[i]);

            // Launch kernel for this chunk
            kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_input + offset, d_output + offset, currentChunkSize,
                                                               args..., offset);

            // Async copy back to host
            cudaMemcpyAsync(output + offset, d_output + offset, outputChunkBytes, cudaMemcpyDeviceToHost, streams[i]);
        }

        // Synchronize and cleanup streams
        for (int i = 0; i < STREAM_COUNT; i++) {
            err = cudaStreamSynchronize(streams[i]);
            if (err != cudaSuccess) {
                // Cleanup everything
                cudaFree(d_input);
                cudaFree(d_output);
                for (int j = 0; j < STREAM_COUNT; j++) {
                    cudaStreamDestroy(streams[j]);
                }
                throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err));
            }
            cudaStreamDestroy(streams[i]);
        }

        // Cleanup device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }

    template<typename InputT, typename OutputT, typename KernelFunc, typename... Args>
    void execute_cuda_kernel(KernelFunc kernel, const InputT* input, OutputT* output, const int n,
                             const int streaming_threshold,
                             Args... args) // Additional kernel arguments (mu, sigma, lambda, stepSize, etc.)
    {
        if (n <= 0) return;

        // Clear any sticky errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA has sticky error: %s\n", cudaGetErrorString(err));
            cudaDeviceReset();
        }

        if (n < streaming_threshold) {
            execute_simple(kernel, input, output, n, args...);
        } else {
            execute_streaming(kernel, input, output, n, args...);
        }
    }

} // namespace fastdist::cuda

#endif // FASTDIST_CUDA_EXECUTOR_CUH
