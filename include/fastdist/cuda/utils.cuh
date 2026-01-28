// include/fastdist/cuda/utils.cuh
#pragma once

namespace fastdist::cuda::utils {
    // Declaration of the C++ dispatcher function (called by the C wrapper)
    void sigmoid_dispatcher(const double* x, double* output, int n);
    void logit_dispatcher(const double* p, double* output, int n);
    void euclidean_distance_dispatcher(const double* x_input, const double* y_input, double* output, const int* strides,
                                       int batch_count);
    void manhattan_distance_dispatcher(const double* x_input, const double* y_input, double* output, const int* strides,
                                       int batch_count);
    void cosine_similarity_dispatcher(const double* x_input, const double* y_input, double* output, const int* strides,
                                      int batch_count);

} // namespace fastdist::cuda::utils
