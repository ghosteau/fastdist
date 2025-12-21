// pybind11 bindings for /src/math/utils.cpp
#include "fastdist/math/utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_chebyshev(py::module_ &m) {
    m.def("chebyshev_bound", &fastdist::math::chebyshev_bound, py::arg("variance"), py::arg("k"),
          R"pbdoc(Manny!
        )pbdoc");
}
