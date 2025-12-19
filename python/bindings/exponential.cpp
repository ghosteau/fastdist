// pybind11 bindings for /src/math/exponential.cpp

#include "fastdist/math/exponential.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_exponential(py::module_ &m) {
    m.def("exponential_pdf_scalar", &fastdist::math::exponential_pdf_scalar, py::arg("x"), py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("exponential_cdf_scalar", &fastdist::math::exponential_cdf_scalar, py::arg("x"), py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("exponential_mean", &fastdist::math::exponential_mean, py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("exponential_variance", &fastdist::math::exponential_variance, py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("exponential_stddev", &fastdist::math::exponential_stddev, py::arg("lambda"),
          R"pbdoc(Manny!
        )pbdoc");

    m.attr("__version__") = "0.0.1";
}
