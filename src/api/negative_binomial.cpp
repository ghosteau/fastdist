// src/api/negative_binomial.cpp
#include <fastdist/math/negative_binomial.h>

extern "C" double fd_negative_binomial_pmf(const int k, const int r, const double p) {
    return fastdist::math::negative_binomial_pmf_scalar(k, r, p);
}

extern "C" double fd_negative_binomial_cdf(const int k, const int r, const double p) {
    return fastdist::math::negative_binomial_cdf_scalar(k, r, p);
}

extern "C" double fd_negative_binomial_mean(const int r, const double p) {
    return fastdist::math::negative_binomial_mean(r, p);
}

extern "C" double fd_negative_binomial_variance(const int r, const double p) {
    return fastdist::math::negative_binomial_variance(r, p);
}

extern "C" double fd_negative_binomial_stddev(const int r, double p) {
    return fastdist::math::negative_binomial_stddev(r, p);
}

extern "C" double fd_negative_binomial_mgf(const double t, const int r, const double p) {
    return fastdist::math::negative_binomial_mgf_scalar(t, r, p);
}

extern "C" double fd_negative_binomial_cgf(const double t, const int r, const double p) {
    return fastdist::math::negative_binomial_cgf_scalar(t, r, p);
}

extern "C" int fd_negative_binomial_sample(const int r, const double p) {
    return fastdist::math::negative_binomial_sample(r, p);
}
