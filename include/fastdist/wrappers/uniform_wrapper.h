// /include/fastdist/wrappers/uniform_wrapper.h
#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace fastdist::math {

    py::array_t<double> uniform_pdf_cpu_wrapper(const py::array_t<double>& x, double a, double b,
                                                double stepSize = 0.0);
    py::array_t<double> uniform_cdf_cpu_wrapper(const py::array_t<double>& x, double a, double b,
                                                double stepSize = 0.0);
    py::array_t<double> uniform_mgf_cpu_wrapper(const py::array_t<double>& t, double a, double b,
                                                double stepSize = 0.0);
    py::array_t<double> uniform_cgf_cpu_wrapper(const py::array_t<double>& t, double a, double b,
                                                double stepSize = 0.0);

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> uniform_pdf_cuda_wrapper(const py::array_t<double>& x, double a, double b,
                                                 double stepSize = 0.0);
    py::array_t<double> uniform_cdf_cuda_wrapper(const py::array_t<double>& x, double a, double b,
                                                 double stepSize = 0.0);
    py::array_t<double> uniform_mgf_cuda_wrapper(const py::array_t<double>& t, double a, double b,
                                                 double stepSize = 0.0);
    py::array_t<double> uniform_cgf_cuda_wrapper(const py::array_t<double>& t, double a, double b,
                                                 double stepSize = 0.0);
#endif

} // namespace fastdist::math
