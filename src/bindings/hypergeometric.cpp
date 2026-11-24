// pybind11 bindings for /src/math/hypergeometric.cpp
#include "fastdist/math/hypergeometric.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_hypergeometric(py::module_ &m) {
    m.def("hypergeometric_logpmf_scalar", &fastdist::math::hypergeometric_logpmf_scalar, py::arg("k"),
          py::arg("population"), py::arg("successes"), py::arg("draws"),
          R"pbdoc(Compute log PMF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_pmf_scalar", &fastdist::math::hypergeometric_pmf_scalar, py::arg("k"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute PMF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_cdf_scalar", &fastdist::math::hypergeometric_cdf_scalar, py::arg("k"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute CDF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_mean", &fastdist::math::hypergeometric_mean, py::arg("population"), py::arg("successes"),
          py::arg("draws"), R"pbdoc(Compute mean of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_variance", &fastdist::math::hypergeometric_variance, py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute variance of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_stddev", &fastdist::math::hypergeometric_stddev, py::arg("population"), py::arg("successes"),
          py::arg("draws"), R"pbdoc(Compute standard deviation of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_mgf_scalar", &fastdist::math::hypergeometric_mgf_scalar, py::arg("t"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute MGF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_cgf_scalar", &fastdist::math::hypergeometric_cgf_scalar, py::arg("t"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute CGF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_sample", &fastdist::math::hypergeometric_sample, py::arg("population"), py::arg("successes"),
          py::arg("draws"), R"pbdoc(Draw random sample from Hypergeometric distribution)pbdoc");
}
