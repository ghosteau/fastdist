// pybind11 bindings for /src/math/geometric.cpp
#include "fastdist/math/geometric.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_geometric(py::module_ &m) {
    m.def("geometric_pmf_scalar", &fastdist::math::geometric_pmf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Compute PMF of geometric distribution)pbdoc");

    m.def("geometric_cdf_scalar", &fastdist::math::geometric_cdf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Compute CDF of geometric distribution)pbdoc");

    m.def("geometric_mean", &fastdist::math::geometric_mean, py::arg("p"),
          R"pbdoc(Compute mean of geometric distribution)pbdoc");

    m.def("geometric_variance", &fastdist::math::geometric_variance, py::arg("p"),
          R"pbdoc(Compute variance of geometric distribution)pbdoc");

    m.def("geometric_stddev", &fastdist::math::geometric_stddev, py::arg("p"),
          R"pbdoc(Compute standard deviation of geometric distribution)pbdoc");

    m.def("geometric_mgf_scalar", &fastdist::math::geometric_mgf_scalar, py::arg("t"), py::arg("p"),
          R"pbdoc(Compute MGF of geometric distribution)pbdoc");

    m.def("geometric_cgf_scalar", &fastdist::math::geometric_cgf_scalar, py::arg("t"), py::arg("p"),
          R"pbdoc(Compute CGF of geometric distribution)pbdoc");

    m.def("geometric_sample", &fastdist::math::geometric_sample, py::arg("p"),
          R"pbdoc(Draw random sample from geometric distribution)pbdoc");
}
