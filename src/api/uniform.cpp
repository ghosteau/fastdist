// src/api/uniform.cpp
#include <fastdist/math/uniform.h>

extern "C" double fd_uniform_pdf(const double x, const double a, const double b) {
    return fastdist::math::uniform_pdf_scalar(x, a, b);
}

extern "C" double fd_uniform_cdf(const double x, const double a, const double b) {
    return fastdist::math::uniform_cdf_scalar(x, a, b);
}

extern "C" double fd_uniform_mean(const double a, const double b) { return fastdist::math::uniform_mean(a, b); }

extern "C" double fd_uniform_variance(const double a, const double b) { return fastdist::math::uniform_variance(a, b); }

extern "C" double fd_uniform_stddev(const double a, const double b) { return fastdist::math::uniform_stddev(a, b); }

extern "C" double fd_uniform_mgf(const double t, const double a, const double b) {
    return fastdist::math::uniform_mgf_scalar(t, a, b);
}

extern "C" double fd_uniform_cgf(const double t, const double a, const double b) {
    return fastdist::math::uniform_cgf_scalar(t, a, b);
}

extern "C" double fd_uniform_sample(const double a, const double b) { return fastdist::math::uniform_sample(a, b); }
