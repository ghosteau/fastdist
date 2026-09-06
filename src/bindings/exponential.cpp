// pybind11 bindings for /src/math/exponential.cpp
#include "fastdist/math/exponential.h"
#include <pybind11/pybind11.h>

#include "wrappers/exponential_wrapper.h"

namespace py = pybind11;

void bind_exponential(py::module_ &m) {
    m.def("exponential_pdf_scalar", &fastdist::math::exponential_pdf_scalar, py::arg("x"), py::arg("lambda_"),
          R"pbdoc(Compute PDF of exponential distribution)pbdoc");

    m.def("exponential_cdf_scalar", &fastdist::math::exponential_cdf_scalar, py::arg("x"), py::arg("lambda_"),
          R"pbdoc(Compute CDF of exponential distribution)pbdoc");

    m.def("exponential_mean", &fastdist::math::exponential_mean, py::arg("lambda_"),
          R"pbdoc(Compute mean of exponential distribution)pbdoc");

    m.def("exponential_variance", &fastdist::math::exponential_variance, py::arg("lambda_"),
          R"pbdoc(Compute variance of exponential distribution)pbdoc");

    m.def("exponential_stddev", &fastdist::math::exponential_stddev, py::arg("lambda_"),
          R"pbdoc(Compute standard deviation of exponential distribution)pbdoc");

    m.def("exponential_mgf_scalar", &fastdist::math::exponential_mgf_scalar, py::arg("t"), py::arg("lambda_"),
          R"pbdoc(Compute MGF of exponential distribution)pbdoc");

    m.def("exponential_cgf_scalar", &fastdist::math::exponential_cgf_scalar, py::arg("t"), py::arg("lambda_"),
          R"pbdoc(Compute CGF of exponential distribution)pbdoc");

    m.def("exponential_sample", &fastdist::math::exponential_sample, py::arg("lambda_"),
          R"pbdoc(Draw random sample from exponential distribution)pbdoc");

    // Batch Functions
    m.def("exponential_pdf_cpu", &fastdist::math::exponential_pdf_cpu_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential PDF on CPU)pbdoc");

    m.def("exponential_cdf_cpu", &fastdist::math::exponential_cdf_cpu_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential CDF on CPU)pbdoc");

    m.def("exponential_mgf_cpu", &fastdist::math::exponential_mgf_cpu_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential MGF on CPU)pbdoc");

    m.def("exponential_cgf_cpu", &fastdist::math::exponential_cgf_cpu_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential CGF on CPU)pbdoc");

#ifdef FASTDIST_ENABLE_CUDA
    // CUDA Functions
    m.def("exponential_pdf_cuda", &fastdist::math::exponential_pdf_cuda_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential PDF using CUDA (GPU))pbdoc");
    m.def("exponential_cdf_cuda", &fastdist::math::exponential_cdf_cuda_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential PDF using CUDA (GPU))pbdoc");
    m.def("exponential_mgf_cuda", &fastdist::math::exponential_mgf_cuda_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential PDF using CUDA (GPU))pbdoc");
    m.def("exponential_cgf_cuda", &fastdist::math::exponential_cgf_cuda_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute exponential PDF using CUDA (GPU))pbdoc");
#endif
}
