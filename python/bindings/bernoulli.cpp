// pybind11 bindings for /src/math/bernoulli.cpp

#include <pybind11/pybind11.h>
#include "fastdist/math/bernoulli.h"

namespace py = pybind11;

PYBIND11_MODULE(fastdist, m) {
    m.def("bernoulli_pmf_scalar",
        &fastdist::math::bernoulli_pmf_scalar,
        py::arg("k"),
        py::arg("p"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_cdf_scalar",
        &fastdist::math::bernoulli_cdf_scalar,
        py::arg("k"),
        py::arg("p"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_mean",
        &fastdist::math::bernoulli_mean,
        py::arg("p"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_variance",
        &fastdist::math::bernoulli_variance,
        py::arg("p"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_stddev",
        &fastdist::math::bernoulli_stddev,
        py::arg("p"),
        R"pbdoc(Manny!
        )pbdoc");

    m.attr("__version__") = "0.0.1";
}