// src/api/binomial.cpp
#include <fastdist/math/binomial.h>

extern "C" double fd_binomial_pmf(int x, int n, double p) {
    return fastdist::math::binomial_pmf_scalar(x, n, p);
}

extern "C" double fd_binomial_logpmf(int x, int n, double p) {
    return fastdist::math::binomial_logpmf_scalar(x, n, p);
}

extern "C" double fd_binomial_cdf(int x, int n, double p) {
    return fastdist::math::binomial_cdf_scalar(x, n, p);
}

extern "C" double fd_binomial_mean(int n, double p) {
    return fastdist::math::binomial_mean(n, p);
}

extern "C" double fd_binomial_variance(int n, double p) {
    return fastdist::math::binomial_variance(n, p);
}

extern "C" double fd_binomial_stddev(int n, double p) {
    return fastdist::math::binomial_stddev(n, p);
}
