// pybind11 bindings for /src/math/exponential.cpp
#include "fastdist/math/exponential.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_exponential(py::module_ &m) {
    m.def("exponential_pdf_scalar", &fastdist::math::exponential_pdf_scalar, py::arg("x"), py::arg("lambda"),
          R"pbdoc(Compute PDF of exponential distribution)pbdoc");

    m.def("exponential_cdf_scalar", &fastdist::math::exponential_cdf_scalar, py::arg("x"), py::arg("lambda"),
          R"pbdoc(Compute CDF of exponential distribution)pbdoc");

    m.def("exponential_mean", &fastdist::math::exponential_mean, py::arg("lambda"),
          R"pbdoc(Compute mean of exponential distribution)pbdoc");

    m.def("exponential_variance", &fastdist::math::exponential_variance, py::arg("lambda"),
          R"pbdoc(Compute variance of exponential distribution)pbdoc");

    m.def("exponential_stddev", &fastdist::math::exponential_stddev, py::arg("lambda"),
          R"pbdoc(Compute standard deviation of exponential distribution)pbdoc");

    m.def("exponential_mgf_scalar", &fastdist::math::exponential_mgf_scalar, py::arg("t"), py::arg("lambda"),
          R"pbdoc(Compute MGF of exponential distribution)pbdoc");

    m.def("exponential_cgf_scalar", &fastdist::math::exponential_cgf_scalar, py::arg("t"), py::arg("lambda"),
          R"pbdoc(Compute CGF of exponential distribution)pbdoc");

    m.def("exponential_sample", &fastdist::math::exponential_sample, py::arg("lambda"),
          R"pbdoc(Draw random sample from exponential distribution)pbdoc");
}
