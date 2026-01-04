// pybind11 bindings for /src/math/discrete_uniform.cpp
#include "fastdist/math/discrete_uniform.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_discrete_uniform(py::module_ &m) {
    m.def("discrete_uniform_pmf_scalar", &fastdist::math::discrete_uniform_pmf_scalar, py::arg("x"), py::arg("a"),
          py::arg("b"), R"pbdoc(Compute PMF of discrete uniform distribution)pbdoc");

    m.def("discrete_uniform_cdf_scalar", &fastdist::math::discrete_uniform_cdf_scalar, py::arg("x"), py::arg("a"),
          py::arg("b"), R"pbdoc(Compute CDF of discrete uniform distribution)pbdoc");

    m.def("discrete_uniform_mean", &fastdist::math::discrete_uniform_mean, py::arg("a"), py::arg("b"),
          R"pbdoc(Compute mean of discrete uniform distribution)pbdoc");

    m.def("discrete_uniform_variance", &fastdist::math::discrete_uniform_variance, py::arg("a"), py::arg("b"),
          R"pbdoc(Compute variance of discrete uniform distribution)pbdoc");

    m.def("discrete_uniform_stddev", &fastdist::math::discrete_uniform_stddev, py::arg("a"), py::arg("b"),
          R"pbdoc(Compute standard deviation of discrete uniform distribution)pbdoc");

    m.def("discrete_uniform_mgf_scalar", &fastdist::math::discrete_uniform_mgf_scalar, py::arg("t"), py::arg("a"),
          py::arg("b"), R"pbdoc(Compute MGF of discrete uniform distribution)pbdoc");

    m.def("discrete_uniform_cgf_scalar", &fastdist::math::discrete_uniform_cgf_scalar, py::arg("t"), py::arg("a"),
          py::arg("b"), R"pbdoc(Compute CGF of discrete uniform distribution)pbdoc");

    m.def("discrete_uniform_sample", &fastdist::math::discrete_uniform_sample, py::arg("a"), py::arg("b"),
          R"pbdoc(Draw random sample from discrete uniform distribution)pbdoc");
}
