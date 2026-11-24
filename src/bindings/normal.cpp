// pybind11 bindings for /src/math/normal.cpp
#include "probforge/math/normal.h"
#include <pybind11/pybind11.h>
#include "probforge/wrappers/normal_wrapper.h"

#ifdef PROBFORGE_ENABLE_CUDA
#include "probforge/cuda/normal.cuh"
#endif

namespace py = pybind11;

void bind_normal(py::module_ &m) {
    m.def("normal_pdf_scalar", &probforge::math::normal_pdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute PDF of normal distribution)pbdoc");

    m.def("normal_logpdf_scalar", &probforge::math::normal_logpdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute log-PDF of normal distribution)pbdoc");

    m.def("normal_cdf_scalar", &probforge::math::normal_cdf_scalar, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute CDF of normal distribution)pbdoc");

    m.def("normal_mean", &probforge::math::normal_mean, py::arg("mu"),
          R"pbdoc(Compute mean of normal distribution)pbdoc");

    m.def("normal_variance", &probforge::math::normal_variance, py::arg("sigma"),
          R"pbdoc(Compute variance of normal distribution)pbdoc");

    m.def("normal_stddev", &probforge::math::normal_stddev, py::arg("sigma"),
          R"pbdoc(Compute standard deviation of normal distribution)pbdoc");

    m.def("normal_mgf_scalar", &probforge::math::normal_mgf_scalar, py::arg("t"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute MGF of normal distribution)pbdoc");

    m.def("normal_cgf_scalar", &probforge::math::normal_cgf_scalar, py::arg("t"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute CGF of normal distribution)pbdoc");

    m.def("normal_sample", &probforge::math::normal_sample, py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Draw random sample from normal distribution)pbdoc");

    m.def("normal_log_sample", &probforge::math::normal_log_sample, py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Draw log-domain random sample from normal distribution)pbdoc");

    m.def("z_score", &probforge::math::z_score, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          R"pbdoc(Compute z-score for normal distribution)pbdoc");

    // Batch Functions
    m.def("normal_pdf_cpu", &probforge::math::normal_pdf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF on CPU)pbdoc");

    m.def("normal_logpdf_cpu", &probforge::math::normal_logpdf_cpu_wrapper, py::arg("x"), py::arg("mu"),
          py::arg("sigma"), py::arg("step_size"), R"pbdoc(Batch compute normal Log PDF on CPU)pbdoc");

    m.def("normal_cdf_cpu", &probforge::math::normal_cdf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal CDF on CPU)pbdoc");

    m.def("normal_mgf_cpu", &probforge::math::normal_mgf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal MGF on CPU)pbdoc");

    m.def("normal_cgf_cpu", &probforge::math::normal_cgf_cpu_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal CGF on CPU)pbdoc");

#ifdef PROBFORGE_ENABLE_CUDA
    // CUDA Functions
    m.def("normal_pdf_cuda", &probforge::math::normal_pdf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_logpdf_cuda", &probforge::math::normal_logpdf_cuda_wrapper, py::arg("x"), py::arg("mu"),
          py::arg("sigma"), py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_cdf_cuda", &probforge::math::normal_cdf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_mgf_cuda", &probforge::math::normal_mgf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
    m.def("normal_cgf_cuda", &probforge::math::normal_cgf_cuda_wrapper, py::arg("x"), py::arg("mu"), py::arg("sigma"),
          py::arg("step_size"), R"pbdoc(Batch compute normal PDF using CUDA (GPU))pbdoc");
#endif
}
