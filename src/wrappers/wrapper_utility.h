#ifndef FASTDIST_WRAPPER_UTILITY_H
#define FASTDIST_WRAPPER_UTILITY_H

#include <cstddef> // size_t
#include <pybind11/numpy.h>
#include <stdexcept> // std::runtime_error
#include <utility> // std::forward
#include <vector>

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/executor.cuh"
#endif

namespace py = pybind11;

namespace fastdist::wrapper {
    // CPU Implementation
    template<typename InputT, typename OutputT, typename BatchFn, typename... Args>
    py::array_t<OutputT> run_cpu_wrapper(BatchFn fn, const py::array_t<InputT>& input, Args&&... args) {
        // Get the array's(input's) information
        const auto buf = input.request();

        if (buf.ndim != 1) {
            throw std::runtime_error("Input array must be 1-dimensional");
        }
        if (buf.strides[0] != static_cast<size_t>(sizeof(InputT))) {
            throw std::runtime_error("Input array must be contiguous");
        }

        // Make a numpy array of x's size
        auto result = py::array_t<OutputT>(buf.size);
        // Get the array's('result') information
        const auto result_buf = result.request();

        // Creates the in and out pointers required for sending and receiving data
        const auto* in_ptr = static_cast<const InputT*>(buf.ptr);
        auto* out_ptr = static_cast<OutputT*>(result_buf.ptr);

        // Gets the size of the array
        const auto n = static_cast<size_t>(buf.shape[0]);

        py::gil_scoped_release release;
        fn(in_ptr, out_ptr, n, std::forward<Args>(args)...);
        return result;
    }

#ifdef FASTDIST_ENABLE_CUDA
    // CUDA Implementation
    template<typename InputT, typename OutputT, typename CudaFn, typename... Args>
    py::array_t<OutputT> run_cuda_wrapper(CudaFn fn, const pybind11::array_t<InputT>& input, Args&&... args) {
        py::gil_scoped_release release;

        const auto buf = input.request();

        if (buf.ndim != 1) {
            throw std::runtime_error("Input arrays must be 1-dimensional");
        }
        if (buf.strides[0] != static_cast<size_t>(sizeof(InputT))) {
            throw std::runtime_error("Input array must be contiguous");
        }

        auto result = py::array_t<OutputT>(buf.size);
        const auto result_buf = result.request();

        const auto* in_ptr = static_cast<const InputT*>(buf.ptr);
        auto* out_ptr = static_cast<OutputT*>(result_buf.ptr);

        const auto n = static_cast<size_t>(buf.shape[0]);

        fn(in_ptr, out_ptr, n, std::forward<Args>(args)...);

        return result;
    }

    template<typename Dispatcher>
    py::array_t<double> run_cuda_distance_wrapper(Dispatcher dispatcher, const py::array_t<double>& x,
                                                  const py::array_t<double>& y) {
        // Request buffers to check dimensions
        auto buf1 = x.request();
        auto buf2 = y.request();

        if (buf1.ndim != 2 || buf2.ndim != 2) {
            throw std::runtime_error("Input arrays must be 2D (batch_count, dims)");
        }

        int batch_count = static_cast<int>(buf1.shape[0]);
        int dims = static_cast<int>(buf1.shape[1]);

        if (buf2.shape[0] != batch_count || buf2.shape[1] != dims) {
            throw std::runtime_error("Input arrays must have the same shape");
        }

        // Prepare host output and strides vector
        py::array_t<double> result(batch_count);
        std::vector<int> strides(batch_count + 1);

        for (int i = 0; i <= batch_count; ++i) {
            strides[i] = i * dims;
        }

        // Call the dispatcher (e.g., euclidean_distance_dispatcher)
        dispatcher(static_cast<const double*>(buf1.ptr), static_cast<const double*>(buf2.ptr),
                   static_cast<double*>(result.request().ptr), strides.data(), batch_count);

        return result;
    }
#endif
} // namespace fastdist::wrapper

#endif // FASTDIST_WRAPPER_UTILITY_H
