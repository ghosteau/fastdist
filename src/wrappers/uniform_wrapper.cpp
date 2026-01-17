// math/uniform_wrapper.cpp

#include "fastdist/wrappers/uniform_wrapper.h"
#include "math/uniform.h"
#include "wrapper_utility.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/uniform.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> uniform_pdf_cpu_wrapper(const py::array_t<double>& x, const double a, const double b,
                                                const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(uniform_pdf_batch, x, a, b, stepSize);
    }

    py::array_t<double> uniform_cdf_cpu_wrapper(const py::array_t<double>& x, const double a, const double b,
                                                const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(uniform_cdf_batch, x, a, b, stepSize);
    }

    py::array_t<double> uniform_mgf_cpu_wrapper(const py::array_t<double>& t, const double a, const double b,
                                                const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(uniform_mgf_batch, t, a, b, stepSize);
    }

    py::array_t<double> uniform_cgf_cpu_wrapper(const py::array_t<double>& t, const double a, const double b,
                                                const double stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(uniform_cgf_batch, t, a, b, stepSize);
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> uniform_pdf_cuda_wrapper(const py::array_t<double>& x, const double a, const double b,
                                                 const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::uniform::uniform_pdf_dispatcher, x,
                                                                   a, b, stepSize);
    }

    py::array_t<double> uniform_cdf_cuda_wrapper(const py::array_t<double>& x, const double a, const double b,
                                                 const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::uniform::uniform_cdf_dispatcher, x,
                                                                   a, b, stepSize);
    }

    py::array_t<double> uniform_mgf_cuda_wrapper(const py::array_t<double>& t, const double a, const double b,
                                                 const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::uniform::uniform_mgf_dispatcher, t,
                                                                   a, b, stepSize);
    }

    py::array_t<double> uniform_cgf_cuda_wrapper(const py::array_t<double>& t, const double a, const double b,
                                                 const double stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::uniform::uniform_cgf_dispatcher, t,
                                                                   a, b, stepSize);
    }

#endif
} // namespace fastdist::math
