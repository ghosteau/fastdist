// math/bernoulli_wrapper.cpp

#include "probforge/wrappers/bernoulli_wrapper.h"
#include "probforge/math/bernoulli.h"
#include "wrapper_utility.h"

#ifdef PROBFORGE_ENABLE_CUDA
#include "probforge/cuda/bernoulli.cuh"
#endif

namespace probforge::math {
    py::array_t<double> bernoulli_pmf_cpu_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<int, double>(bernoulli_pmf_batch, k, p, stepSize);
    }

    py::array_t<double> bernoulli_cdf_cpu_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<int, double>(bernoulli_cdf_batch, k, p, stepSize);
    }

    py::array_t<double> bernoulli_mgf_cpu_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(bernoulli_mgf_batch, t, p, stepSize);
    }

    py::array_t<double> bernoulli_cgf_cpu_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(bernoulli_cgf_batch, t, p, stepSize);
    }

#ifdef PROBFORGE_ENABLE_CUDA
    py::array_t<double> bernoulli_pmf_cuda_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<int, double>(probforge::cuda::bernoulli::bernoulli_pmf_dispatcher,
                                                                 k, p, stepSize);
    }

    py::array_t<double> bernoulli_cdf_cuda_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<int, double>(probforge::cuda::bernoulli::bernoulli_cdf_dispatcher,
                                                                 k, p, stepSize);
    }

    py::array_t<double> bernoulli_mgf_cuda_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(
                probforge::cuda::bernoulli::bernoulli_mgf_dispatcher, t, p, stepSize);
    }

    py::array_t<double> bernoulli_cgf_cuda_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(
                probforge::cuda::bernoulli::bernoulli_cgf_dispatcher, t, p, stepSize);
    }
#endif
} // namespace probforge::math
