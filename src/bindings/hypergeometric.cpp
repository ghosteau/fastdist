// pybind11 bindings for /src/math/hypergeometric.cpp
#include "probforge/math/hypergeometric.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_hypergeometric(py::module_ &m) {
    m.def("hypergeometric_logpmf_scalar", &probforge::math::hypergeometric_logpmf_scalar, py::arg("k"),
          py::arg("population"), py::arg("successes"), py::arg("draws"),
          R"pbdoc(Compute log PMF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_pmf_scalar", &probforge::math::hypergeometric_pmf_scalar, py::arg("k"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute PMF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_cdf_scalar", &probforge::math::hypergeometric_cdf_scalar, py::arg("k"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute CDF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_mean", &probforge::math::hypergeometric_mean, py::arg("population"), py::arg("successes"),
          py::arg("draws"), R"pbdoc(Compute mean of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_variance", &probforge::math::hypergeometric_variance, py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute variance of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_stddev", &probforge::math::hypergeometric_stddev, py::arg("population"), py::arg("successes"),
          py::arg("draws"), R"pbdoc(Compute standard deviation of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_mgf_scalar", &probforge::math::hypergeometric_mgf_scalar, py::arg("t"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute MGF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_cgf_scalar", &probforge::math::hypergeometric_cgf_scalar, py::arg("t"), py::arg("population"),
          py::arg("successes"), py::arg("draws"), R"pbdoc(Compute CGF of Hypergeometric distribution)pbdoc");

    m.def("hypergeometric_sample", &probforge::math::hypergeometric_sample, py::arg("population"), py::arg("successes"),
          py::arg("draws"), R"pbdoc(Draw random sample from Hypergeometric distribution)pbdoc");
}
