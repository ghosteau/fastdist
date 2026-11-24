// math/poisson_wrapper.cpp

#include "probforge/wrappers/poisson_wrapper.h"
#include "math/poisson.h"
#include "wrapper_utility.h"

#ifdef PROBFORGE_ENABLE_CUDA
#include "probforge/cuda/poisson.cuh"
#endif

namespace probforge::math {
    py::array_t<double> poisson_pmf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(poisson_pmf_batch, x, lambda, stepSize);
    }

    py::array_t<double> poisson_cdf_cpu_wrapper(const py::array_t<double>& x, const double lambda, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(poisson_cdf_batch, x, lambda, stepSize);
    }

    py::array_t<double> poisson_mgf_cpu_wrapper(const py::array_t<double>& t, const double lambda, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(poisson_mgf_batch, t, lambda, stepSize);
    }

    py::array_t<double> poisson_cgf_cpu_wrapper(const py::array_t<double>& t, const double lambda, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(poisson_cgf_batch, t, lambda, stepSize);
    }

#ifdef PROBFORGE_ENABLE_CUDA
    py::array_t<double> poisson_pmf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::poisson::poisson_pmf_dispatcher, x,
                                                                    lambda, stepSize);
    }

    py::array_t<double> poisson_cdf_cuda_wrapper(const py::array_t<double>& x, const double lambda,
                                                 const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::poisson::poisson_cdf_dispatcher, x,
                                                                    lambda, stepSize);
    }

    py::array_t<double> poisson_mgf_cuda_wrapper(const py::array_t<double>& t, const double lambda,
                                                 const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::poisson::poisson_mgf_dispatcher, t,
                                                                    lambda, stepSize);
    }

    py::array_t<double> poisson_cgf_cuda_wrapper(const py::array_t<double>& t, const double lambda,
                                                 const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::poisson::poisson_cgf_dispatcher, t,
                                                                    lambda, stepSize);
    }

#endif
} // namespace probforge::math
