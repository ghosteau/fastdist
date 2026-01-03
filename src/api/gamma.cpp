// src/api/gamma.cpp
#include <fastdist/math/gamma.h>

extern "C" double fd_gamma_pdf(const double x, const double alpha, const double theta) {
    return fastdist::math::gamma_pdf_scalar(x, alpha, theta);
}

extern "C" double fd_gamma_cdf(const double x, const double alpha, const double theta) {
    return fastdist::math::gamma_cdf_scalar(x, alpha, theta);
}

extern "C" double fd_gamma_mean(const double alpha, const double theta) {
    return fastdist::math::gamma_mean(alpha, theta);
}

extern "C" double fd_gamma_variance(const double alpha, const double theta) {
    return fastdist::math::gamma_variance(alpha, theta);
}

extern "C" double fd_gamma_stddev(const double alpha, const double theta) {
    return fastdist::math::gamma_stddev(alpha, theta);
}

extern "C" double fd_gamma_mgf(const double t, const double alpha, const double theta) {
    return fastdist::math::gamma_mgf_scalar(t, alpha, theta);
}

extern "C" double fd_gamma_cgf(const double t, const double alpha, const double theta) {
    return fastdist::math::gamma_cgf_scalar(t, alpha, theta);
}

extern "C" double fd_gamma_sample(const double alpha, const double theta) {
    return fastdist::math::gamma_sample(alpha, theta);
}
