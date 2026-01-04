// pybind11 bindings for /src/math/binomial.cpp
#include "fastdist/math/binomial.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_binomial(py::module_ &m) {
    m.def("binomial_logpmf_scalar", &fastdist::math::binomial_logpmf_scalar, py::arg("x"), py::arg("n"), py::arg("p"),
          R"pbdoc(Compute log PMF of Binomial distribution)pbdoc");

    m.def("binomial_pmf_scalar", &fastdist::math::binomial_pmf_scalar, py::arg("x"), py::arg("n"), py::arg("p"),
          R"pbdoc(Compute PMF of Binomial distribution)pbdoc");

    m.def("binomial_cdf_scalar", &fastdist::math::binomial_cdf_scalar, py::arg("x"), py::arg("n"), py::arg("p"),
          R"pbdoc(Compute CDF of Binomial distribution)pbdoc");

    m.def("binomial_mean", &fastdist::math::binomial_mean, py::arg("n"), py::arg("p"),
          R"pbdoc(Compute mean of Binomial distribution)pbdoc");

    m.def("binomial_variance", &fastdist::math::binomial_variance, py::arg("n"), py::arg("p"),
          R"pbdoc(Compute variance of Binomial distribution)pbdoc");

    m.def("binomial_stddev", &fastdist::math::binomial_stddev, py::arg("n"), py::arg("p"),
          R"pbdoc(Compute standard deviation of Binomial distribution)pbdoc");

    m.def("binomial_mgf_scalar", &fastdist::math::binomial_mgf_scalar, py::arg("t"), py::arg("n"), py::arg("p"),
          R"pbdoc(Compute MGF of Binomial distribution)pbdoc");

    m.def("binomial_cgf_scalar", &fastdist::math::binomial_cgf_scalar, py::arg("t"), py::arg("n"), py::arg("p"),
          R"pbdoc(Compute CGF of Binomial distribution)pbdoc");

    m.def("binomial_sample", &fastdist::math::binomial_sample, py::arg("n"), py::arg("p"),
          R"pbdoc(Draw random sample from Binomial distribution)pbdoc");
}
