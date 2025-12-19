// pybind11 bindings for /src/math/poisson.cpp
#include "fastdist/math/poisson.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_poisson(py::module_ &m) {
    m.def("poisson_pmf_scalar", &fastdist::math::poisson_pmf_scalar, py::arg("k"), py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("poisson_cdf_scalar", &fastdist::math::poisson_cdf_scalar, py::arg("k"), py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("poisson_mean", &fastdist::math::poisson_mean, py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("poisson_variance", &fastdist::math::poisson_variance, py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("poisson_stddev", &fastdist::math::poisson_stddev, py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.attr("__version__") = "0.0.1";
}
