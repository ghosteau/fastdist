// src/api/binomial.cpp
#include <fastdist/math/binomial.h>

extern "C" double fd_binomial_pmf(const int x, const int n, const double p) {
    return fastdist::math::binomial_pmf_scalar(x, n, p);
}

extern "C" double fd_binomial_logpmf(const int x, const int n, const double p) {
    return fastdist::math::binomial_logpmf_scalar(x, n, p);
}

extern "C" double fd_binomial_cdf(const int x, const int n, const double p) {
    return fastdist::math::binomial_cdf_scalar(x, n, p);
}

extern "C" double fd_binomial_mean(const int n, const double p) { return fastdist::math::binomial_mean(n, p); }

extern "C" double fd_binomial_variance(const int n, const double p) { return fastdist::math::binomial_variance(n, p); }

extern "C" double fd_binomial_stddev(const int n, const double p) { return fastdist::math::binomial_stddev(n, p); }
