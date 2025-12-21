// pybind11 bindings for /src/math/geometric.cpp
#include "fastdist/math/geometric.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_geometric(py::module_ &m) {
    m.def("geometric_pmf_scalar", &fastdist::math::geometric_pmf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("geometric_cdf_scalar", &fastdist::math::geometric_cdf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("geometric_mean", &fastdist::math::geometric_mean, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("geometric_variance", &fastdist::math::geometric_variance, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("geometric_stddev", &fastdist::math::geometric_stddev, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");
}
