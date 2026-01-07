// include/fastdist/cuda/normal.cuh
#pragma once

namespace fastdist::cuda::normal {
    // Declaration of the C++ dispatcher function (called by the C wrapper)
    void normal_pdf_dispatcher(const double* x, double* output, int n, double mu, double sigma, double stepSize);
    void normal_logpdf_dispatcher(const double* x, double* output, int n, double mu, double sigma, double stepSize);
    void normal_cdf_dispatcher(const double* x, double* output, int n, double mu, double sigma, double stepSize);
    void normal_mgf_dispatcher(const double* t, double* output, int n, double mu, double sigma, double stepSize);
    void normal_cgf_dispatcher(const double* t, double* output, int n, double mu, double sigma, double stepSize);

} // namespace fastdist::cuda::normal
