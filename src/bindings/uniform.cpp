// pybind11 bindings for /src/math/uniform.cpp
#include "fastdist/math/uniform.h"
#include "pybind11/pybind11.h"
#include "wrappers/uniform_wrapper.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/uniform.cuh"
#endif

namespace py = pybind11;

void bind_uniform(py::module_ &m) {
    m.def("uniform_pdf_scalar", &fastdist::math::uniform_pdf_scalar, py::arg("x"), py::arg("a"), py::arg("b"),
          R"pbdoc(Compute PDF of continuous uniform distribution)pbdoc");

    m.def("uniform_cdf_scalar", &fastdist::math::uniform_cdf_scalar, py::arg("x"), py::arg("a"), py::arg("b"),
          R"pbdoc(Compute CDF of continuous uniform distribution)pbdoc");

    m.def("uniform_mean", &fastdist::math::uniform_mean, py::arg("a"), py::arg("b"),
          R"pbdoc(Compute mean of continuous uniform distribution)pbdoc");

    m.def("uniform_variance", &fastdist::math::uniform_variance, py::arg("a"), py::arg("b"),
          R"pbdoc(Compute variance of continuous uniform distribution)pbdoc");

    m.def("uniform_stddev", &fastdist::math::uniform_stddev, py::arg("a"), py::arg("b"),
          R"pbdoc(Compute standard deviation of continuous uniform distribution)pbdoc");

    m.def("uniform_mgf_scalar", &fastdist::math::uniform_mgf_scalar, py::arg("t"), py::arg("a"), py::arg("b"),
          R"pbdoc(Compute MGF of continuous uniform distribution)pbdoc");

    m.def("uniform_cgf_scalar", &fastdist::math::uniform_cgf_scalar, py::arg("t"), py::arg("a"), py::arg("b"),
          R"pbdoc(Compute CGF of continuous uniform distribution)pbdoc");

    m.def("uniform_sample", &fastdist::math::uniform_sample, py::arg("a"), py::arg("b"),
          R"pbdoc(Draw random sample from continuous uniform distribution)pbdoc");

    // Batch Functions
    m.def("uniform_pdf_cpu", &fastdist::math::uniform_pdf_cpu_wrapper, py::arg("x"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform PMF on CPU)pbdoc");

    m.def("uniform_cdf_cpu", &fastdist::math::uniform_cdf_cpu_wrapper, py::arg("x"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform CDF on CPU)pbdoc");

    m.def("uniform_mgf_cpu", &fastdist::math::uniform_mgf_cpu_wrapper, py::arg("t"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform MGF on CPU)pbdoc");

    m.def("uniform_cgf_cpu", &fastdist::math::uniform_cgf_cpu_wrapper, py::arg("t"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform CGF on CPU)pbdoc");

#ifdef FASTDIST_ENABLE_CUDA
    // CUDA Functions
    m.def("uniform_pdf_cuda", &fastdist::math::uniform_pdf_cuda_wrapper, py::arg("x"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform PMF using CUDA (GPU))pbdoc");
    m.def("uniform_cdf_cuda", &fastdist::math::uniform_cdf_cuda_wrapper, py::arg("x"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform CDF using CUDA (GPU))pbdoc");
    m.def("uniform_mgf_cuda", &fastdist::math::uniform_mgf_cuda_wrapper, py::arg("t"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform MGF using CUDA (GPU))pbdoc");
    m.def("uniform_cgf_cuda", &fastdist::math::uniform_cgf_cuda_wrapper, py::arg("t"), py::arg("a"), py::arg("b"),
          py::arg("step_size"), R"pbdoc(Batch compute uniform CGF using CUDA (GPU))pbdoc");
#endif
}
