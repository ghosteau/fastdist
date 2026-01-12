// pybind11 bindings for /src/math/poisson.cpp
#include "fastdist/cuda/poisson.cuh"
#include <pybind11/pybind11.h>
#include "fastdist/math/poisson.h"
#include "fastdist/wrappers/poisson_wrapper.h"

namespace py = pybind11;

void bind_poisson(py::module_ &m) {
    m.def("poisson_pmf_scalar", &fastdist::math::poisson_pmf_scalar, py::arg("k"), py::arg("lambda"),
          R"pbdoc(Compute PMF of Poisson distribution)pbdoc");

    m.def("poisson_cdf_scalar", &fastdist::math::poisson_cdf_scalar, py::arg("k"), py::arg("lambda"),
          R"pbdoc(Compute CDF of Poisson distribution)pbdoc");

    m.def("poisson_mean", &fastdist::math::poisson_mean, py::arg("lambda"),
          R"pbdoc(Compute mean of Poisson distribution)pbdoc");

    m.def("poisson_variance", &fastdist::math::poisson_variance, py::arg("lambda"),
          R"pbdoc(Compute variance of Poisson distribution)pbdoc");

    m.def("poisson_stddev", &fastdist::math::poisson_stddev, py::arg("lambda"),
          R"pbdoc(Compute standard deviation of Poisson distribution)pbdoc");

    m.def("poisson_mgf_scalar", &fastdist::math::poisson_mgf_scalar, py::arg("t"), py::arg("lambda"),
          R"pbdoc(Compute MGF of Poisson distribution)pbdoc");

    m.def("poisson_cgf_scalar", &fastdist::math::poisson_cgf_scalar, py::arg("t"), py::arg("lambda"),
          R"pbdoc(Compute CGF of Poisson distribution)pbdoc");

    m.def("poisson_sample", &fastdist::math::poisson_sample, py::arg("lambda"),
          R"pbdoc(Draw random sample from Poisson distribution)pbdoc");

    // Batch Functions
    m.def("poisson_pmf_cpu", &fastdist::math::poisson_pmf_cpu_wrapper, py::arg("x"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson PMF on CPU)pbdoc");

    m.def("poisson_cdf_cpu", &fastdist::math::poisson_cdf_cpu_wrapper, py::arg("x"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CDF on CPU)pbdoc");

    m.def("poisson_mgf_cpu", &fastdist::math::poisson_mgf_cpu_wrapper, py::arg("t"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson MGF on CPU)pbdoc");

    m.def("poisson_cgf_cpu", &fastdist::math::poisson_cgf_cpu_wrapper, py::arg("t"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CGF on CPU)pbdoc");

#ifdef FASTDIST_ENABLE_CUDA
    // CUDA Functions
    m.def("poisson_pmf_cuda", &fastdist::math::poisson_pmf_cuda_wrapper, py::arg("x"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson PMF using CUDA (GPU))pbdoc");
    m.def("poisson_cdf_cuda", &fastdist::math::poisson_cdf_cuda_wrapper, py::arg("x"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CDF using CUDA (GPU))pbdoc");
    m.def("poisson_mgf_cuda", &fastdist::math::poisson_mgf_cuda_wrapper, py::arg("t"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson MGF using CUDA (GPU))pbdoc");
    m.def("poisson_cgf_cuda", &fastdist::math::poisson_cgf_cuda_wrapper, py::arg("t"), py::arg("lambda"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CGF using CUDA (GPU))pbdoc");
#endif
}
