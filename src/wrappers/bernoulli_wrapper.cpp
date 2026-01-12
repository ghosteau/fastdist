// math/bernoulli_wrapper.cpp

#include "fastdist/wrappers/bernoulli_wrapper.h"
#include "fastdist/math/bernoulli.h"
#include "wrapper_utility.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/bernoulli.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> bernoulli_pmf_cpu_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<int, double>(bernoulli_pmf_batch, k, p, stepSize);
    }

    py::array_t<double> bernoulli_cdf_cpu_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<int, double>(bernoulli_cdf_batch, k, p, stepSize);
    }

    py::array_t<double> bernoulli_mgf_cpu_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(bernoulli_mgf_batch, t, p, stepSize);
    }

    py::array_t<double> bernoulli_cgf_cpu_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(bernoulli_cgf_batch, t, p, stepSize);
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> bernoulli_pmf_cuda_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<int, double>(fastdist::cuda::bernoulli::bernoulli_pmf_dispatcher, k,
                                                                p, stepSize);
    }

    py::array_t<double> bernoulli_cdf_cuda_wrapper(const py::array_t<int>& k, const double p, const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<int, double>(fastdist::cuda::bernoulli::bernoulli_cdf_dispatcher, k,
                                                                p, stepSize);
    }

    py::array_t<double> bernoulli_mgf_cuda_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::bernoulli::bernoulli_mgf_dispatcher,
                                                                   t, p, stepSize);
    }

    py::array_t<double> bernoulli_cgf_cuda_wrapper(const py::array_t<double>& t, const double p, const int stepSize) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::bernoulli::bernoulli_cgf_dispatcher,
                                                                   t, p, stepSize);
    }
#endif
} // namespace fastdist::math
