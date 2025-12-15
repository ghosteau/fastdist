// src/api/normal.cpp
#include <fastdist/math/normal.h>

extern "C" double fd_normal_pdf(double x, double mu, double sigma) {
    return fastdist::math::normal_pdf_scalar(x, mu, sigma);
}

extern "C" double fd_normal_cdf(double x, double mu, double sigma) {
    return fastdist::math::normal_cdf_scalar(x, mu, sigma);
}

extern "C" double fd_normal_mean(double mu) {
    return fastdist::math::normal_mean(mu);
}

extern "C" double fd_normal_variance(double sigma) {
    return fastdist::math::normal_variance(sigma);
}

extern "C" double fd_normal_stddev(double sigma) {
    return fastdist::math::normal_stddev(sigma);
}
