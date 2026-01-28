// /include/fastdist/math/utils_wrapper.h
#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace fastdist::math {
    py::array_t<double> sigmoid_cpu_wrapper(const py::array_t<double>& x);
    py::array_t<double> logit_cpu_wrapper(const py::array_t<double>& p);

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> sigmoid_cuda_wrapper(const py::array_t<double>& x);
    py::array_t<double> logit_cuda_wrapper(const py::array_t<double>& p);
    py::array_t<double> euclidean_distance_cuda_wrapper(const py::array_t<double>& x, const py::array_t<double>& y);
    py::array_t<double> manhattan_distance_cuda_wrapper(const py::array_t<double>& x, const py::array_t<double>& y);
    py::array_t<double> cosine_similarity_cuda_wrapper(const py::array_t<double>& x, const py::array_t<double>& y);
#endif
} // namespace fastdist::math
