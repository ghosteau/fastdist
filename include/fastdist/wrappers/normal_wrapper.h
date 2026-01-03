// /include/fastdist/math/normal_bindings.h
#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace fastdist::math {
    py::array_t<double> normal_pdf_cpu_wrapper(const py::array_t<double>& x, double mu, double sigma);

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> normal_pdf_cuda_wrapper(const py::array_t<double>& x, double mu, double sigma);
#endif
} // namespace fastdist::math
