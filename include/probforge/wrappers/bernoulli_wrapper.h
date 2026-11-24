// /include/probforge/math/bernoulli.h
#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace probforge::math {
    py::array_t<double> bernoulli_pmf_cpu_wrapper(const py::array_t<int>& k, double p, int stepSize = 0);
    py::array_t<double> bernoulli_cdf_cpu_wrapper(const py::array_t<int>& k, double p, int stepSize = 0);
    py::array_t<double> bernoulli_mgf_cpu_wrapper(const py::array_t<double>& t, double p, int stepSize = 0);
    py::array_t<double> bernoulli_cgf_cpu_wrapper(const py::array_t<double>& t, double p, int stepSize = 0);

#ifdef PROBFORGE_ENABLE_CUDA
    py::array_t<double> bernoulli_pmf_cuda_wrapper(const py::array_t<int>& k, double p, int stepSize = 0);
    py::array_t<double> bernoulli_cdf_cuda_wrapper(const py::array_t<int>& k, double p, int stepSize = 0);
    py::array_t<double> bernoulli_mgf_cuda_wrapper(const py::array_t<double>& t, double p, int stepSize = 0);
    py::array_t<double> bernoulli_cgf_cuda_wrapper(const py::array_t<double>& t, double p, int stepSize = 0);

#endif
} // namespace probforge::math
