// src/api/beta.cpp
#include <fastdist/math/beta.h>

extern "C" double fd_beta_pdf(const double x, const double alpha, const double beta) {
    return fastdist::math::beta_pdf_scalar(x, alpha, beta);
}

extern "C" double fd_beta_cdf(const double x, const double alpha, const double beta) {
    return fastdist::math::beta_cdf_scalar(x, alpha, beta);
}

extern "C" double fd_beta_mean(const double alpha, const double beta) { return fastdist::math::beta_mean(alpha, beta); }

extern "C" double fd_beta_variance(const double alpha, const double beta) {
    return fastdist::math::beta_variance(alpha, beta);
}

extern "C" double fd_beta_stddev(const double alpha, const double beta) {
    return fastdist::math::beta_stddev(alpha, beta);
}

extern "C" double fd_beta_sample(const double alpha, const double beta) {
    return fastdist::math::beta_sample(alpha, beta);
}
