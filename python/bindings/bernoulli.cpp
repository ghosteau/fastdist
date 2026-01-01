// pybind11 bindings for /src/math/bernoulli.cpp
#include "fastdist/math/bernoulli.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_bernoulli(py::module_ &m) {
    m.def("bernoulli_pmf_scalar", &fastdist::math::bernoulli_pmf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_cdf_scalar", &fastdist::math::bernoulli_cdf_scalar, py::arg("k"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_mean", &fastdist::math::bernoulli_mean, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_variance", &fastdist::math::bernoulli_variance, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_stddev", &fastdist::math::bernoulli_stddev, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_mgf_scalar", &fastdist::math::bernoulli_mgf_scalar, py::arg("t"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_cgf_scalar", &fastdist::math::bernoulli_cgf_scalar, py::arg("t"), py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bernoulli_sample", &fastdist::math::bernoulli_sample, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");
}
