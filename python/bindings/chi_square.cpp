// pybind11 bindings for /src/math/chi_square.cpp
#include "fastdist/math/chi_square.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_chi_square(py::module_ &m) {
    m.def("chi_square_pdf_scalar", &fastdist::math::chi_square_pdf_scalar, py::arg("x"), py::arg("k"),
          R"pbdoc(Manny!
      )pbdoc");

    m.def("chi_square_cdf_scalar", &fastdist::math::chi_square_cdf_scalar, py::arg("x"), py::arg("k"),
          R"pbdoc(Manny!
      )pbdoc");

    m.def("chi_square_mean", &fastdist::math::chi_square_mean, py::arg("k"),
          R"pbdoc(Manny!
      )pbdoc");

    m.def("chi_square_variance", &fastdist::math::chi_square_variance, py::arg("k"),
          R"pbdoc(Manny!
    )pbdoc");

    m.def("chi_square_stddev", &fastdist::math::chi_square_stddev, py::arg("k"),
          R"pbdoc(Manny!
      )pbdoc");

    m.def("chi_square_mgf_scalar", &fastdist::math::chi_square_mgf_scalar, py::arg("t"), py::arg("k"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("chi_square_cgf_scalar", &fastdist::math::chi_square_cgf_scalar, py::arg("t"), py::arg("k"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("chi_square_sample", &fastdist::math::chi_square_sample, py::arg("k"),
          R"pbdoc(Manny!
    )pbdoc");
}
