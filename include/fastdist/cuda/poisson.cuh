// include/fastdist/cuda/poisson.cuh
#pragma once

namespace fastdist::cuda::poisson {
    // Declaration of the C++ dispatcher function (called by the C wrapper)
    void poisson_pmf_dispatcher(const double* x, double* output, int n, double lambda_, int stepSize);
    void poisson_cdf_dispatcher(const double* x, double* output, int n, double lambda_, int stepSize);
    void poisson_mgf_dispatcher(const double* t, double* output, int n, double lambda_, int stepSize);
    void poisson_cgf_dispatcher(const double* t, double* output, int n, double lambda_, int stepSize);

} // namespace fastdist::cuda::poisson
