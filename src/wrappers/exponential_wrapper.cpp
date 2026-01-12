// math/exponential_wrapper.cpp

#include "fastdist/wrappers/exponential_wrapper.h"
#include "fastdist/math/exponential.h"
#include "wrapper_utility.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/exponential.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> exponential_pdf_cpu_wrapper(const py::array_t<double>& x, const double lambda,
                                                    const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(exponential_pdf_batch, x, lambda, stepSize);
    }

    py::array_t<double> exponential_cdf_cpu_wrapper(const py::array_t<double>& x, const double lambda,
                                                    const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(exponential_cdf_batch, x, lambda, stepSize);
    }

    py::array_t<double> exponential_mgf_cpu_wrapper(const py::array_t<double>& t, const double lambda,
                                                    const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(exponential_mgf_batch, t, lambda, stepSize);
    }

    py::array_t<double> exponential_cgf_cpu_wrapper(const py::array_t<double>& t, const double lambda,
                                                    const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(exponential_cgf_batch, t, lambda, stepSize);
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> exponential_pdf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                     const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(
                fastdist::cuda::exponential::exponential_pdf_dispatcher, x, lambda, stepSize);
    }

    py::array_t<double> exponential_cdf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                     const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(
                fastdist::cuda::exponential::exponential_cdf_dispatcher, x, lambda, stepSize);
    }

    py::array_t<double> exponential_mgf_cuda_wrapper(const py::array_t<double>& t, const double lambda,
                                                     const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(
                fastdist::cuda::exponential::exponential_mgf_dispatcher, t, lambda, stepSize);
    }

    py::array_t<double> exponential_cgf_cuda_wrapper(const py::array_t<double>& t, const double lambda,
                                                     const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(
                fastdist::cuda::exponential::exponential_cgf_dispatcher, t, lambda, stepSize);
    }
#endif
} // namespace fastdist::math
