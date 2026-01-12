// include/fastdist/cuda/exponential.cuh
#pragma once

namespace fastdist::cuda::exponential {
    // Declaration of the C++ dispatcher function (called by the C wrapper)
    void exponential_pdf_dispatcher(const double* x, double* output, int n, double lambda, double stepSize);
    void exponential_cdf_dispatcher(const double* x, double* output, int n, double lambda, double stepSize);
    void exponential_mgf_dispatcher(const double* t, double* output, int n, double lambda, double stepSize);
    void exponential_cgf_dispatcher(const double* t, double* output, int n, double lambda, double stepSize);

} // namespace fastdist::cuda::exponential
