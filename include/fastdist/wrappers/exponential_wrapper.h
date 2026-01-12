// /include/fastdist/math/exponential.h
#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace fastdist::math {
    py::array_t<double> exponential_pdf_cpu_wrapper(const py::array_t<double>& x, double lambda, double stepSize = 0);
    py::array_t<double> exponential_cdf_cpu_wrapper(const py::array_t<double>& x, double lambda, double stepSize = 0);
    py::array_t<double> exponential_mgf_cpu_wrapper(const py::array_t<double>& t, double lambda, double stepSize = 0);
    py::array_t<double> exponential_cgf_cpu_wrapper(const py::array_t<double>& t, double lambda, double stepSize = 0);

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> exponential_pdf_cuda_wrapper(const py::array_t<double>& x, double lambda, double stepSize = 0);
    py::array_t<double> exponential_cdf_cuda_wrapper(const py::array_t<double>& x, double lambda, double stepSize = 0);
    py::array_t<double> exponential_mgf_cuda_wrapper(const py::array_t<double>& t, double lambda, double stepSize = 0);
    py::array_t<double> exponential_cgf_cuda_wrapper(const py::array_t<double>& t, double lambda, double stepSize = 0);
#endif
} // namespace fastdist::math
