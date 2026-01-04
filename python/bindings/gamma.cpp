// pybind11 bindings for /src/math/gamma.cpp
#include "fastdist/math/gamma.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_gamma(py::module_ &m) {
    m.def("gamma_pdf_scalar", &fastdist::math::gamma_pdf_scalar, py::arg("x"), py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Compute PDF of Gamma distribution)pbdoc");

    m.def("gamma_cdf_scalar", &fastdist::math::gamma_cdf_scalar, py::arg("x"), py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Compute CDF of Gamma distribution)pbdoc");

    m.def("gamma_mean", &fastdist::math::gamma_mean, py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Compute mean of Gamma distribution)pbdoc");

    m.def("gamma_variance", &fastdist::math::gamma_variance, py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Compute variance of Gamma distribution)pbdoc");

    m.def("gamma_stddev", &fastdist::math::gamma_stddev, py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Compute standard deviation of Gamma distribution)pbdoc");

    m.def("gamma_mgf_scalar", &fastdist::math::gamma_mgf_scalar, py::arg("t"), py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Compute MGF of Gamma distribution)pbdoc");

    m.def("gamma_cgf_scalar", &fastdist::math::gamma_cgf_scalar, py::arg("t"), py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Compute CGF of Gamma distribution)pbdoc");

    m.def("gamma_sample", &fastdist::math::gamma_sample, py::arg("alpha"), py::arg("theta"),
          R"pbdoc(Draw random sample from Gamma distribution)pbdoc");
}
