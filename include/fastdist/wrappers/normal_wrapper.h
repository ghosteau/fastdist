// /include/fastdist/math/normal_wrapper.h
#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace fastdist::math {
    py::array_t<double> normal_pdf_cpu_wrapper(const py::array_t<double>& x, double mu, double sigma,
                                               double stepSize = 0);
    py::array_t<double> normal_logpdf_cpu_wrapper(const py::array_t<double>& x, double mu, double sigma,
                                                  double stepSize = 0);
    py::array_t<double> normal_cdf_cpu_wrapper(const py::array_t<double>& x, double mu, double sigma,
                                               double stepSize = 0);
    py::array_t<double> normal_mgf_cpu_wrapper(const py::array_t<double>& t, double mu, double sigma,
                                               double stepSize = 0);
    py::array_t<double> normal_cgf_cpu_wrapper(const py::array_t<double>& t, double mu, double sigma,
                                               double stepSize = 0);

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> normal_pdf_cuda_wrapper(const py::array_t<double>& x, double mu, double sigma,
                                                double stepSize = 0);
    py::array_t<double> normal_logpdf_cuda_wrapper(const py::array_t<double>& x, double mu, double sigma,
                                                   double stepSize = 0);
    py::array_t<double> normal_cdf_cuda_wrapper(const py::array_t<double>& x, double mu, double sigma,
                                                double stepSize = 0);
    py::array_t<double> normal_mgf_cuda_wrapper(const py::array_t<double>& t, double mu, double sigma,
                                                double stepSize = 0);
    py::array_t<double> normal_cgf_cuda_wrapper(const py::array_t<double>& t, double mu, double sigma,
                                                double stepSize = 0);

#endif
} // namespace fastdist::math
