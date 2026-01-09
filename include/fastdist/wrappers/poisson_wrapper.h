// /include/fastdist/math/poisson_wrapper.h
#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace fastdist::math {

    py::array_t<double> poisson_pmf_cpu_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);
    py::array_t<double> poisson_cdf_cpu_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);
    py::array_t<double> poisson_mgf_cpu_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);
    py::array_t<double> poisson_cgf_cpu_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> poisson_pmf_cuda_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);
    py::array_t<double> poisson_cdf_cuda_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);
    py::array_t<double> poisson_mgf_cuda_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);
    py::array_t<double> poisson_cgf_cuda_wrapper(const py::array_t<double>& x, double lambda, int stepSize = 0);
#endif

} // namespace fastdist::math
