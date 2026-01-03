// include/fastdist/cuda/normal.cuh
#pragma once

namespace fastdist::cuda::normal {
    // Declaration of the C++ dispatcher function (called by the C wrapper)
    void normal_pdf_dispatcher(const double* x, double* output, int n, double mu, double sigma);

} // namespace fastdist::cuda::normal
