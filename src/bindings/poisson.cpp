// pybind11 bindings for /src/math/poisson.cpp
#include "probforge/cuda/poisson.cuh"
#include <pybind11/pybind11.h>
#include "probforge/math/poisson.h"
#include "probforge/wrappers/poisson_wrapper.h"


#ifdef PROBFORGE_ENABLE_CUDA
#include "probforge/cuda/poisson.cuh"
#endif

namespace py = pybind11;

void bind_poisson(py::module_ &m) {
    m.def("poisson_pmf_scalar", &probforge::math::poisson_pmf_scalar, py::arg("k"), py::arg("lambda_"),
          R"pbdoc(Compute PMF of Poisson distribution)pbdoc");

    m.def("poisson_cdf_scalar", &probforge::math::poisson_cdf_scalar, py::arg("k"), py::arg("lambda_"),
          R"pbdoc(Compute CDF of Poisson distribution)pbdoc");

    m.def("poisson_mean", &probforge::math::poisson_mean, py::arg("lambda_"),
          R"pbdoc(Compute mean of Poisson distribution)pbdoc");

    m.def("poisson_variance", &probforge::math::poisson_variance, py::arg("lambda_"),
          R"pbdoc(Compute variance of Poisson distribution)pbdoc");

    m.def("poisson_stddev", &probforge::math::poisson_stddev, py::arg("lambda_"),
          R"pbdoc(Compute standard deviation of Poisson distribution)pbdoc");

    m.def("poisson_mgf_scalar", &probforge::math::poisson_mgf_scalar, py::arg("t"), py::arg("lambda_"),
          R"pbdoc(Compute MGF of Poisson distribution)pbdoc");

    m.def("poisson_cgf_scalar", &probforge::math::poisson_cgf_scalar, py::arg("t"), py::arg("lambda_"),
          R"pbdoc(Compute CGF of Poisson distribution)pbdoc");

    m.def("poisson_sample", &probforge::math::poisson_sample, py::arg("lambda_"),
          R"pbdoc(Draw random sample from Poisson distribution)pbdoc");

    // Batch Functions
    m.def("poisson_pmf_cpu", &probforge::math::poisson_pmf_cpu_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson PMF on CPU)pbdoc");

    m.def("poisson_cdf_cpu", &probforge::math::poisson_cdf_cpu_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CDF on CPU)pbdoc");

    m.def("poisson_mgf_cpu", &probforge::math::poisson_mgf_cpu_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson MGF on CPU)pbdoc");

    m.def("poisson_cgf_cpu", &probforge::math::poisson_cgf_cpu_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CGF on CPU)pbdoc");

#ifdef PROBFORGE_ENABLE_CUDA
    // CUDA Functions
    m.def("poisson_pmf_cuda", &probforge::math::poisson_pmf_cuda_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson PMF using CUDA (GPU))pbdoc");
    m.def("poisson_cdf_cuda", &probforge::math::poisson_cdf_cuda_wrapper, py::arg("x"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CDF using CUDA (GPU))pbdoc");
    m.def("poisson_mgf_cuda", &probforge::math::poisson_mgf_cuda_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson MGF using CUDA (GPU))pbdoc");
    m.def("poisson_cgf_cuda", &probforge::math::poisson_cgf_cuda_wrapper, py::arg("t"), py::arg("lambda_"),
          py::arg("step_size"), R"pbdoc(Batch compute poisson CGF using CUDA (GPU))pbdoc");
#endif
}
