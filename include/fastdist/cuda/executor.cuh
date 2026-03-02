// src/cuda/executor.cuh
#ifndef FASTDIST_EXECUTOR_CUH
#define FASTDIST_EXECUTOR_CUH

#include <algorithm>
#include <cuda_runtime.h>
#include <memory> // For std::unique_ptr
#include <stdexcept>
#include <string>

namespace fastdist::cuda {
    struct StreamingThresholds {
        static constexpr int SIMPLE_MATH = 1000000; // exp, log, sqrt, etc.
        static constexpr int COMPLEX_MATH = 500000; // trigonometric, special functions
        static constexpr int VERY_COMPLEX = 100000; // iterative algorithms, numerical integration
        static constexpr int MEMORY_BOUND = 800000; // operations limited by memory bandwidth
    };

    template<typename T>
    struct CudaBuffer {
        T* d_ptr = nullptr;
        size_t count = 0;

        // Constructor
        explicit CudaBuffer(const size_t n) : count(n) {
            const cudaError_t err = cudaMalloc(&d_ptr, n * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate GPU memory: " + std::string(cudaGetErrorString(err)));
            }
        }

        // Destructor
        ~CudaBuffer() {
            if (d_ptr) cudaFree(d_ptr);
        }

        // Disable copy semantics
        CudaBuffer(const CudaBuffer&) = delete;
        CudaBuffer& operator=(const CudaBuffer&) = delete;
    };

    template<typename T, typename U>
    struct DeviceContext {
        CudaBuffer<T> dev_in;
        CudaBuffer<U> dev_out;
        size_t capacity;

        DeviceContext(const size_t n) : dev_in(n), dev_out(n), capacity(n) {}
    };

    template<typename T, typename U>
    static DeviceContext<T, U>& get_context(size_t n) {
        static std::unique_ptr<DeviceContext<T, U>> instance = nullptr;

        if (!instance || instance->capacity < n) {
            instance = std::make_unique<DeviceContext<T, U>>(n);
        }
        return *instance;
    }

    template<typename InputT, typename OutputT, typename KernelFunc, typename... Args>
    static void execute_simple(KernelFunc kernel, const InputT* input, OutputT* output, CudaBuffer<InputT>& dev_in,
                               CudaBuffer<OutputT>& dev_out, // Device buffers for input and output
                               const int n, Args... args) {
#ifdef __CUDACC__
        if (dev_in.count < n || dev_out.count < n) {
            throw std::runtime_error("GPU Buffer too small for request size n");
        }

        // Use a static stream to avoid the overhead of creating/destroying one
        static cudaStream_t simple_stream;
        static bool stream_init = false;
        if (!stream_init) {
            cudaStreamCreate(&simple_stream);
            stream_init = true;
        }

        const size_t inputSize = n * sizeof(InputT);
        const size_t outputSize = n * sizeof(OutputT);

        // Move data to GPU
        cudaMemcpyAsync(dev_in.d_ptr, input, inputSize, cudaMemcpyHostToDevice, simple_stream);

        // Computation
        constexpr int threads = 256;
        int blocks = (n + threads - 1) / threads;
        kernel<<<blocks, threads, 0, simple_stream>>>(dev_in.d_ptr, dev_out.d_ptr, n, args..., 0); // Pass offset as 0

        // Retrieve data from GPU
        cudaMemcpyAsync(output, dev_out.d_ptr, outputSize, cudaMemcpyDeviceToHost, simple_stream);

        cudaStreamSynchronize(simple_stream);
#endif
    }

    template<typename InputT, typename OutputT, typename KernelFunc, typename... Args>
    static void execute_streaming(KernelFunc kernel, const InputT* input, OutputT* output, CudaBuffer<InputT>& dev_in,
                                  CudaBuffer<OutputT>& dev_out, const int n, Args... args) {
#ifdef __CUDACC__
        static constexpr int STREAM_COUNT = 4;

        // Static streams
        static cudaStream_t streams[STREAM_COUNT];
        static bool streams_initialized = false;
        if (!streams_initialized) {
            for (cudaStream_t& stream: streams) cudaStreamCreate(&stream);
            streams_initialized = true;
        }

        const int chunkSize = (n + STREAM_COUNT - 1) / STREAM_COUNT;

        for (int i = 0; i < STREAM_COUNT; i++) {
            const int offset = i * chunkSize;
            const int currentChunkSize = std::min(chunkSize, n - offset);
            if (currentChunkSize <= 0) break;
            constexpr int threads = 256;
            const int blocks = (currentChunkSize + threads - 1) / threads;

            cudaMemcpyAsync(dev_in.d_ptr + offset, input + offset, currentChunkSize * sizeof(InputT),
                            cudaMemcpyHostToDevice, streams[i]);

            kernel<<<blocks, threads, 0, streams[i]>>>(dev_in.d_ptr + offset, dev_out.d_ptr + offset, currentChunkSize,
                                                       args..., offset);

            cudaMemcpyAsync(output + offset, dev_out.d_ptr + offset, currentChunkSize * sizeof(OutputT),
                            cudaMemcpyDeviceToHost, streams[i]);
        }

        for (cudaStream_t& stream: streams) {
            cudaStreamSynchronize(stream);
        }
#endif
    }

    template<typename InputT, typename OutputT, typename KernelFunc, typename... Args>
    void execute_cuda_kernel(KernelFunc kernel, const InputT* input, OutputT* output, CudaBuffer<InputT>& dev_in,
                             CudaBuffer<OutputT>& dev_out, const int n, const int streaming_threshold,
                             Args... args) // Additional kernel arguments (mu, sigma, lambda, stepSize, etc.)
    {
        if (n <= 0) return;

        // Clear any sticky errors
        if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            fprintf(stderr, "CUDA has sticky error: %s\n", cudaGetErrorString(err));
            cudaDeviceReset();
        }

        if (n < streaming_threshold) {
            execute_simple(kernel, input, output, dev_in, dev_out, n, args...);
        } else {
            execute_streaming(kernel, input, output, dev_in, dev_out, n, args...);
        }
    }
} // namespace fastdist::cuda

#endif // FASTDIST_CUDA_EXECUTOR_CUH
