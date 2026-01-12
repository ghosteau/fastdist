// math/poisson_wrapper.cpp

#include "fastdist/wrappers/poisson_wrapper.h"
#include "math/poisson.h"
#include "wrapper_utility.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/poisson.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> poisson_pmf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(poisson_pmf_batch, x, lambda, stepSize);
    }

    py::array_t<double> poisson_cdf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(poisson_cdf_batch, x, lambda, stepSize);
    }

    py::array_t<double> poisson_mgf_cpu_wrapper(const py::array_t<double>& t, const double lambda, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(poisson_mgf_batch, t, lambda, stepSize);
    }

    py::array_t<double> poisson_cgf_cpu_wrapper(const py::array_t<double>& t, const double lambda, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(poisson_cgf_batch, t, lambda, stepSize);
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> poisson_pmf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::poisson::poisson_pmf_dispatcher, x,
                                                                   lambda, stepSize);
    }

    py::array_t<double> poisson_cdf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::poisson::poisson_cdf_dispatcher, x,
                                                                   lambda, stepSize);
    }

    py::array_t<double> poisson_mgf_cuda_wrapper(const py::array_t<double>& t, const double lambda,
                                                 const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::poisson::poisson_mgf_dispatcher, t,
                                                                   lambda, stepSize);
    }

    py::array_t<double> poisson_cgf_cuda_wrapper(const py::array_t<double>& t, const double lambda,
                                                 const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::poisson::poisson_cgf_dispatcher, t,
                                                                   lambda, stepSize);
    }

#endif
} // namespace fastdist::math
