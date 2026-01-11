// math/bernoulli_wrapper.cpp

#include "fastdist/wrappers/bernoulli_wrapper.h"
#include "fastdist/math/bernoulli.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/bernoulli.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> bernoulli_pmf_cpu_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info k_buf = k.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(k_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* k_ptr = static_cast<int*>(k_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(k_buf.shape[0]);
        bernoulli_pmf_batch(k_ptr, out_ptr, n, p, stepSize);

        return result;
    }

    py::array_t<double> bernoulli_cdf_cpu_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        const pybind11::buffer_info k_buf = k.request();
        const auto result = py::array_t<double>(k_buf.size);
        const pybind11::buffer_info result_buf = result.request();

        const auto* k_ptr = static_cast<const int*>(k_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(k_buf.shape[0]);
        bernoulli_cdf_batch(k_ptr, out_ptr, n, p, stepSize);

        return result;
    }

    py::array_t<double> bernoulli_mgf_cpu_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        const pybind11::buffer_info t_buf = t.request();
        const auto result = py::array_t<double>(t_buf.size);
        const pybind11::buffer_info result_buf = result.request();

        const auto* t_ptr = static_cast<const double*>(t_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(t_buf.shape[0]);
        bernoulli_mgf_batch(t_ptr, out_ptr, n, p, stepSize);

        return result;
    }

    py::array_t<double> bernoulli_cgf_cpu_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        const pybind11::buffer_info t_buf = t.request();
        const auto result = py::array_t<double>(t_buf.size);
        const pybind11::buffer_info result_buf = result.request();

        const auto* t_ptr = static_cast<const double*>(t_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(t_buf.shape[0]);
        bernoulli_cgf_batch(t_ptr, out_ptr, n, p, stepSize);

        return result;
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> bernoulli_pmf_cuda_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        const pybind11::buffer_info k_buf = k.request();
        auto result = py::array_t<int>(k_buf.size);
        const auto result_buf = result.request();

        const auto* k_ptr = static_cast<int*>(k_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(k_buf.shape[0]);
        fastdist::cuda::bernoulli::bernoulli_pmf_dispatcher(k_ptr, out_ptr, n, p, stepSize);

        return result;
    }

    py::array_t<double> bernoulli_cdf_cuda_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        const auto k_buf = k.request();
        auto result = py::array_t<double>(k_buf.size);
        const auto result_buf = result.request();

        const auto* k_ptr = static_cast<const int*>(k_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(k_buf.shape[0]);
        fastdist::cuda::bernoulli::bernoulli_cdf_dispatcher(k_ptr, out_ptr, n, p, stepSize);

        return result;
    }

    py::array_t<double> bernoulli_mgf_cuda_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        const auto t_buf = t.request();
        auto result = py::array_t<double>(t_buf.size);
        const auto result_buf = result.request();

        const auto* t_ptr = static_cast<const double*>(t_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(t_buf.shape[0]);
        fastdist::cuda::bernoulli::bernoulli_mgf_dispatcher(t_ptr, out_ptr, n, p, stepSize);

        return result;
    }

    py::array_t<double> bernoulli_cgf_cuda_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        const auto t_buf = t.request();
        auto result = py::array_t<double>(t_buf.size);
        const auto result_buf = result.request();

        const auto* t_ptr = static_cast<const double*>(t_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(t_buf.shape[0]);
        fastdist::cuda::bernoulli::bernoulli_cgf_dispatcher(t_ptr, out_ptr, n, p, stepSize);

        return result;
    }
#endif
} // namespace fastdist::math
