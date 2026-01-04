// pybind11 bindings for /src/math/poisson.cpp
#include "fastdist/math/poisson.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_poisson(py::module_ &m) {
    m.def("poisson_pmf_scalar", &fastdist::math::poisson_pmf_scalar, py::arg("k"), py::arg("lambda"),
          R"pbdoc(Compute PMF of Poisson distribution)pbdoc");

    m.def("poisson_cdf_scalar", &fastdist::math::poisson_cdf_scalar, py::arg("k"), py::arg("lambda"),
          R"pbdoc(Compute CDF of Poisson distribution)pbdoc");

    m.def("poisson_mean", &fastdist::math::poisson_mean, py::arg("lambda"),
          R"pbdoc(Compute mean of Poisson distribution)pbdoc");

    m.def("poisson_variance", &fastdist::math::poisson_variance, py::arg("lambda"),
          R"pbdoc(Compute variance of Poisson distribution)pbdoc");

    m.def("poisson_stddev", &fastdist::math::poisson_stddev, py::arg("lambda"),
          R"pbdoc(Compute standard deviation of Poisson distribution)pbdoc");

    m.def("poisson_mgf_scalar", &fastdist::math::poisson_mgf_scalar, py::arg("t"), py::arg("lambda"),
          R"pbdoc(Compute MGF of Poisson distribution)pbdoc");

    m.def("poisson_cgf_scalar", &fastdist::math::poisson_cgf_scalar, py::arg("t"), py::arg("lambda"),
          R"pbdoc(Compute CGF of Poisson distribution)pbdoc");

    m.def("poisson_sample", &fastdist::math::poisson_sample, py::arg("lambda"),
          R"pbdoc(Draw random sample from Poisson distribution)pbdoc");
}
