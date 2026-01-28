// math/wrapper.cpp

#include "fastdist/wrappers/utils_wrapper.h"
#include "math/utils.h"
#include "wrapper_utility.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/utils.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> sigmoid_cpu_wrapper(const py::array_t<double>& x) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(sigmoid_cpu, x);
    }

    py::array_t<double> logit_cpu_wrapper(const py::array_t<double>& p) {
        return fastdist::wrapper::run_cpu_wrapper<double, double>(logit_cpu, p);
    }

#ifdef FASTDIST_ENABLE_CUDA

    py::array_t<double> sigmoid_cuda_wrapper(const py::array_t<double>& x) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::utils::sigmoid_dispatcher, x);
    }

    py::array_t<double> logit_cuda_wrapper(const py::array_t<double>& p) {
        return fastdist::wrapper::run_cuda_wrapper<double, double>(fastdist::cuda::utils::logit_dispatcher, p);
    }

    py::array_t<double> euclidean_distance_cuda_wrapper(const py::array_t<double>& x, const py::array_t<double>& y) {
        return fastdist::wrapper::run_cuda_distance_wrapper(fastdist::cuda::utils::euclidean_distance_dispatcher, x, y);
    }

    py::array_t<double> manhattan_distance_cuda_wrapper(const py::array_t<double>& x, const py::array_t<double>& y) {
        return fastdist::wrapper::run_cuda_distance_wrapper(fastdist::cuda::utils::manhattan_distance_dispatcher, x, y);
    }

    py::array_t<double> cosine_similarity_cuda_wrapper(const py::array_t<double>& x, const py::array_t<double>& y) {
        return fastdist::wrapper::run_cuda_distance_wrapper(fastdist::cuda::utils::cosine_similarity_dispatcher, x, y);
    }

#endif
} // namespace fastdist::math
