// include/fastdist/cuda/bernoulli.cuh
#pragma once

namespace fastdist::cuda::bernoulli {
    // Declaration of the C++ dispatcher function (called by the C wrapper)
    void bernoulli_pmf_dispatcher(const int* k, double* output, int n, double p, int stepSize);
    void bernoulli_cdf_dispatcher(const int* k, double* output, int n, double p, int stepSize);
    void bernoulli_mgf_dispatcher(const double* t, double* output, int n, double p, int stepSize);
    void bernoulli_cgf_dispatcher(const double* t, double* output, int n, double p, int stepSize);

} // namespace fastdist::cuda::bernoulli
