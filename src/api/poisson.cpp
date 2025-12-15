// src/api/poisson.cpp
#include <fastdist/math/poisson.h>

extern "C" double fd_poisson_pmf(double x, double lambda) {
    return fastdist::math::poisson_pmf_scalar(x, lambda);
}

extern "C" double fd_poisson_cmf(double x, double lambda) {
    return fastdist::math::poisson_cdf_scalar(x, lambda);
}

extern "C" double fd_poisson_mean(double lambda) {
    return fastdist::math::poisson_mean(lambda);
}

extern "C" double fd_poisson_variance(double lambda) {
    return fastdist::math::poisson_variance(lambda);
}

extern "C" double fd_poisson_stddev(double lambda) {
    return fastdist::math::poisson_stddev(lambda);
}
