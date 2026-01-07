// math/normal_wrapper.cpp

#include "../../include/fastdist/wrappers/normal_wrapper.h"
#include "fastdist/math/normal.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/normal.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> normal_pdf_cpu_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                               double stepSize) {
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
        normal_pdf_batch(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_logpdf_cpu_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                  double stepSize) {
        const pybind11::buffer_info x_buf = x.request();
        const auto result = py::array_t<double>(x_buf.size);
        const pybind11::buffer_info result_buf = result.request();
        const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(x_buf.shape[0]);
        normal_logpdf_batch(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_cdf_cpu_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                               double stepSize) {
        const pybind11::buffer_info x_buf = x.request();
        const auto result = py::array_t<double>(x_buf.size);
        const pybind11::buffer_info result_buf = result.request();

        const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(x_buf.shape[0]);
        normal_cdf_batch(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_mgf_cpu_wrapper(const py::array_t<double>& t, const double mu, const double sigma,
                                               double stepSize) {
        const pybind11::buffer_info t_buf = t.request();
        const auto result = py::array_t<double>(t_buf.size);
        const pybind11::buffer_info result_buf = result.request();

        const auto* t_ptr = static_cast<const double*>(t_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(t_buf.shape[0]);
        normal_mgf_batch(t_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_cgf_cpu_wrapper(const py::array_t<double>& t, const double mu, const double sigma,
                                               double stepSize) {
        const pybind11::buffer_info t_buf = t.request();
        const auto result = py::array_t<double>(t_buf.size);
        const pybind11::buffer_info result_buf = result.request();

        const auto* t_ptr = static_cast<const double*>(t_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const size_t n = static_cast<int>(t_buf.shape[0]);
        normal_cgf_batch(t_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> normal_pdf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                const double stepSize) {
        const auto x_buf = x.request();
        auto result = py::array_t<double>(x_buf.size);
        const auto result_buf = result.request();

        const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const int n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::normal::normal_pdf_dispatcher(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_logpdf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                   const double stepSize) {
        const auto x_buf = x.request();
        auto result = py::array_t<double>(x_buf.size);
        const auto result_buf = result.request();

        const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const int n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::normal::normal_logpdf_dispatcher(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_cdf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                const double stepSize) {
        const auto x_buf = x.request();
        auto result = py::array_t<double>(x_buf.size);
        const auto result_buf = result.request();

        const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const int n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::normal::normal_cdf_dispatcher(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_mgf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                const double stepSize) {
        const auto x_buf = x.request();
        auto result = py::array_t<double>(x_buf.size);
        const auto result_buf = result.request();

        const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const int n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::normal::normal_mgf_dispatcher(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }

    py::array_t<double> normal_cgf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                const double stepSize) {
        const auto x_buf = x.request();
        auto result = py::array_t<double>(x_buf.size);
        const auto result_buf = result.request();

        const auto* x_ptr = static_cast<const double*>(x_buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        const int n = static_cast<int>(x_buf.shape[0]);
        fastdist::cuda::normal::normal_cgf_dispatcher(x_ptr, out_ptr, n, mu, sigma, stepSize);

        return result;
    }
#endif
} // namespace fastdist::math
