// pybind11 bindings for /src/math/negative_binomial.cpp
#include "fastdist/math/negative_binomial.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_negative_binomial(py::module_ &m) {
    m.def("negative_binomial_pmf_scalar", &fastdist::math::negative_binomial_pmf_scalar, py::arg("k"), py::arg("r"),
          py::arg("p"), R"pbdoc(Compute PMF of negative binomial distribution)pbdoc");

    m.def("negative_binomial_cdf_scalar", &fastdist::math::negative_binomial_cdf_scalar, py::arg("k"), py::arg("r"),
          py::arg("p"), R"pbdoc(Compute CDF of negative binomial distribution)pbdoc");

    m.def("negative_binomial_mean", &fastdist::math::negative_binomial_mean, py::arg("r"), py::arg("p"),
          R"pbdoc(Compute mean of negative binomial distribution)pbdoc");

    m.def("negative_binomial_variance", &fastdist::math::negative_binomial_variance, py::arg("r"), py::arg("p"),
          R"pbdoc(Compute variance of negative binomial distribution)pbdoc");

    m.def("negative_binomial_stddev", &fastdist::math::negative_binomial_stddev, py::arg("r"), py::arg("p"),
          R"pbdoc(Compute standard deviation of negative binomial distribution)pbdoc");

    m.def("negative_binomial_mgf_scalar", &fastdist::math::negative_binomial_mgf_scalar, py::arg("t"), py::arg("r"),
          py::arg("p"), R"pbdoc(Compute MGF of negative binomial distribution)pbdoc");

    m.def("negative_binomial_cgf_scalar", &fastdist::math::negative_binomial_cgf_scalar, py::arg("t"), py::arg("r"),
          py::arg("p"), R"pbdoc(Compute CGF of negative binomial distribution)pbdoc");

    m.def("negative_binomial_sample", &fastdist::math::negative_binomial_sample, py::arg("r"), py::arg("p"),
          R"pbdoc(Draw random sample from negative binomial distribution)pbdoc");
}
