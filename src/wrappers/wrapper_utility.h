#ifndef FASTDIST_WRAPPER_UTILITY_H
#define FASTDIST_WRAPPER_UTILITY_H

#include <cstddef> // size_t
#include <pybind11/numpy.h>
#include <utility> // std::forward

namespace fastdist::wrapper {
    // CPU Implementation
    template<typename InputT, typename OutputT, typename BatchFn, typename... Args>
    pybind11::array_t<OutputT> run_cpu_wrapper(BatchFn fn, const pybind11::array_t<InputT>& input, Args&&... args) {
        // Get the array's(input's) information
        const auto buf = input.request();
        // Make a numpy array of x's size
        auto result = pybind11::array_t<OutputT>(buf.size);
        // Get the array's('result') information
        const auto result_buf = result.request();

        // Creates the in and out pointers required for sending and recieving data
        const auto* in_ptr = static_cast<const InputT*>(buf.ptr);
        auto* out_ptr = static_cast<OutputT*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<size_t>(buf.shape[0]);

        fn(in_ptr, out_ptr, n, std::forward<Args>(args)...);
        return result;
    }

#ifdef FASTDIST_ENABLE_CUDA
    // CUDA Implementation
    template<typename InputT, typename OutputT, typename CudaFn, typename... Args>
    pybind11::array_t<OutputT> run_cuda_wrapper(CudaFn fn, const pybind11::array_t<InputT>& input, Args&&... args) {
        const auto buf = input.request();
        auto result = pybind11::array_t<OutputT>(buf.size);
        const auto result_buf = result.request();

        const auto* in_ptr = static_cast<const InputT*>(buf.ptr);
        auto* out_ptr = static_cast<OutputT*>(result_buf.ptr);

        const size_t n = static_cast<size_t>(buf.shape[0]);

        fn(in_ptr, out_ptr, n, std::forward<Args>(args)...);

        return result;
    }
#endif
} // namespace fastdist::wrapper

#endif // FASTDIST_WRAPPER_UTILITY_H
