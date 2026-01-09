// math/poisson_wrapper.cpp

#include "fastdist/wrappers/poisson_wrapper.h"
#include "math/poisson.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/poisson.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> poisson_pmf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        poisson_pmf_batch(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

    py::array_t<double> poisson_cdf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        poisson_cdf_batch(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

    py::array_t<double> poisson_mgf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        poisson_mgf_batch(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

    py::array_t<double> poisson_cgf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        poisson_cgf_batch(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> poisson_pmf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::poisson::poisson_pmf_dispatcher(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

    py::array_t<double> poisson_cdf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::poisson::poisson_cdf_dispatcher(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

    py::array_t<double> poisson_mgf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::poisson::poisson_mgf_dispatcher(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

    py::array_t<double> poisson_cgf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        // Get the array's('x') information
        const pybind11::buffer_info x_buf = x.request();
        // Make a numpy array of x's size
        auto result = py::array_t<double>(x_buf.size);
        // Get the array's('result') information
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Gets the size of the array
        const size_t n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::poisson::poisson_cgf_dispatcher(x_ptr, out_ptr, n, lambda, stepSize);

        return result;
    }

#endif
} // namespace fastdist::math
