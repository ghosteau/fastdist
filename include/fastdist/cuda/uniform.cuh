// include/fastdist/cuda/uniform.cuh
#pragma once

namespace fastdist::cuda::uniform {
    // Declaration of the C++ dispatcher function (called by the C wrapper)
    void uniform_pdf_dispatcher(const double* x, double* output, int n, double a, double b, double stepSize);
    void uniform_cdf_dispatcher(const double* x, double* output, int n, double a, double b, double stepSize);
    void uniform_mgf_dispatcher(const double* t, double* output, int n, double a, double b, double stepSize);
    void uniform_cgf_dispatcher(const double* t, double* output, int n, double a, double b, double stepSize);

} // namespace fastdist::cuda::uniform
