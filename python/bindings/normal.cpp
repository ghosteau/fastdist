// pybind11 bindings for /src/math/normal.cpp

#include <pybind11/pybind11.h>
#include "fastdist/math/normal.h"

namespace py = pybind11;

PYBIND11_MODULE(fastdist, m) {
    m.def("normal_pdf_scalar",
        &fastdist::math::normal_pdf_scalar,
        py::arg("x"),
        py::arg("mu"),
        py::arg("sigma"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_logpdf_scalar",
        &fastdist::math::normal_logpdf_scalar,
        py::arg("x"),
        py::arg("mu"),
        py::arg("sigma"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_cdf_scalar",
        &fastdist::math::normal_cdf_scalar,
        py::arg("x"),
        py::arg("mu"),
        py::arg("sigma"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_mean",
        &fastdist::math::normal_mean,
        py::arg("mu"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_variance",
        &fastdist::math::normal_variance,
        py::arg("sigma"),
        R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_stddev",
        &fastdist::math::normal_stddev,
        py::arg("sigma"),
        R"pbdoc(Manny!
        )pbdoc");

    m.attr("__version__") = "0.0.1";
}