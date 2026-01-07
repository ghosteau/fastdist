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
          R"pbdoc(Compute PDF of normal distribution)pbdoc");

    m.def("normal_logpdf_scalar", &fastdist::math::normal_logpdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute log-PDF of normal distribution)pbdoc");

    m.def("normal_cdf_scalar", &fastdist::math::normal_cdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute CDF of normal distribution)pbdoc");

    m.def("normal_mean", &fastdist::math::normal_mean, py::arg("mu"),
          R"pbdoc(Compute mean of normal distribution)pbdoc");

    m.def("normal_variance", &fastdist::math::normal_variance, py::arg("sigma"),
          R"pbdoc(Compute variance of normal distribution)pbdoc");

    m.def("normal_stddev", &fastdist::math::normal_stddev, py::arg("sigma"),
          R"pbdoc(Compute standard deviation of normal distribution)pbdoc");

    m.def("normal_mgf_scalar", &fastdist::math::normal_mgf_scalar, py::arg("t"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute MGF of normal distribution)pbdoc");

    m.def("normal_cgf_scalar", &fastdist::math::normal_cgf_scalar, py::arg("t"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute CGF of normal distribution)pbdoc");

    m.def("normal_sample", &fastdist::math::normal_sample, py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Draw random sample from normal distribution)pbdoc");

    m.def("normal_log_sample", &fastdist::math::normal_log_sample, py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Draw log-domain random sample from normal distribution)pbdoc");

    m.def("z_score", &fastdist::math::z_score, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute z-score for normal distribution)pbdoc");

    // Batch Functions
    m.def("normal_pdf_cpu", &fastdist::math::normal_pdf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF on CPU)pbdoc");

    m.def("normal_logpdf_cpu", &fastdist::math::normal_logpdf_cpu_wrapper, py::arg("x"), py::arg("mu"),
          py::arg("sigma"), py::arg("step_size"), R"pbdoc(Batch compute normal Log PDF on CPU)pbdoc");

    m.def("normal_cdf_cpu", &fastdist::math::normal_cdf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal CDF on CPU)pbdoc");

    m.def("normal_mgf_cpu", &fastdist::math::normal_mgf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal MGF on CPU)pbdoc");

    m.def("normal_cgf_cpu", &fastdist::math::normal_cgf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal CGF on CPU)pbdoc");

#ifdef FASTDIST_ENABLE_CUDA
    // CUDA Functions
    m.def("normal_pdf_cuda", &fastdist::math::normal_pdf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_logpdf_cuda", &fastdist::math::normal_logpdf_cuda_wrapper, py::arg("x"), py::arg("mu"),
          py::arg("sigma"), py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_cdf_cuda", &fastdist::math::normal_cdf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_mgf_cuda", &fastdist::math::normal_mgf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_cgf_cuda", &fastdist::math::normal_cgf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
#endif
}
