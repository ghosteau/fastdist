// pybind11 bindings for /src/math/normal.cpp
#include "fastdist/cuda/normal.cuh"
#include <pybind11/pybind11.h>
#include "../../include/fastdist/wrappers/normal_wrapper.h"
#include "fastdist/math/normal.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/normal.cuh"
#endif

namespace py = pybind11;

void bind_normal(py::module_ &m) {
    m.def("normal_pdf_scalar", &fastdist::math::normal_pdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_logpdf_scalar", &fastdist::math::normal_logpdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_cdf_scalar", &fastdist::math::normal_cdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_mean", &fastdist::math::normal_mean, py::arg("mu"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_variance", &fastdist::math::normal_variance, py::arg("sigma"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("normal_stddev", &fastdist::math::normal_stddev, py::arg("sigma"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("z_score", &fastdist::math::z_score, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Manny!
        )pbdoc");

    // Batch Functions -
    m.def("normal_pdf_cpu", &fastdist::math::normal_pdf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Manny!
        )pbdoc");

#ifdef FASTDIST_ENABLE_CUDA
    m.def("normal_pdf_cuda", &fastdist::math::normal_pdf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          "Batch Compute Normal PDF using CUDA (GPU)");
#endif
}
