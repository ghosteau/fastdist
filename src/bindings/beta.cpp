// pybind11 bindings for /src/math/beta.cpp
#include "probforge/math/beta.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_beta(py::module_ &m) {
    m.def("beta_pdf_scalar", &probforge::math::beta_pdf_scalar, py::arg("x"), py::arg("alpha"), py::arg("beta"),
          R"pbdoc(Compute PDF of Beta distribution)pbdoc");

    m.def("beta_cdf_scalar", &probforge::math::beta_cdf_scalar, py::arg("x"), py::arg("alpha"), py::arg("beta"),
          R"pbdoc(Compute CDF of Beta distribution)pbdoc");

    m.def("beta_mean", &probforge::math::beta_mean, py::arg("alpha"), py::arg("beta"),
          R"pbdoc(Compute mean of Beta distribution)pbdoc");

    m.def("beta_variance", &probforge::math::beta_variance, py::arg("alpha"), py::arg("beta"),
          R"pbdoc(Compute variance of Beta distribution)pbdoc");

    m.def("beta_stddev", &probforge::math::beta_stddev, py::arg("alpha"), py::arg("beta"),
          R"pbdoc(Compute standard deviation of Beta distribution)pbdoc");

    m.def("beta_sample", &probforge::math::beta_sample, py::arg("alpha"), py::arg("beta"),
          R"pbdoc(Draw random sample from Beta distribution)pbdoc");
}
