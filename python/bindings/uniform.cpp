#include "fastdist/math/uniform.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_uniform(py::module_ &m) {
    m.def("uniform_pdf_scalar", &fastdist::math::uniform_pdf_scalar, py::arg("x"), py::arg("a"), py::arg("b"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("uniform_cdf_scalar", &fastdist::math::uniform_cdf_scalar, py::arg("x"), py::arg("a"), py::arg("b"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("uniform_mean", &fastdist::math::uniform_mean, py::arg("a"), py::arg("b"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("uniform_variance", &fastdist::math::uniform_variance, py::arg("a"), py::arg("b"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("uniform_stddev", &fastdist::math::uniform_stddev, py::arg("a"), py::arg("b"),
          R"pbdoc(Manny!
        )pbdoc");
}
