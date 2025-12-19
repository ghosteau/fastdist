// pybind11 bindings for /src/math/binomial.cpp
#include "fastdist/math/binomial.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_binomial(py::module_ &m) {
    m.def("binomial_pmf_scalar", &fastdist::math::binomial_pmf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("binomial_cdf_scalar", &fastdist::math::binomial_cdf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("binomial_mean", &fastdist::math::binomial_mean, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("binomial_variance", &fastdist::math::binomial_variance, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("binomial_stddev", &fastdist::math::binomial_stddev, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.attr("__version__") = "0.0.1";
}
