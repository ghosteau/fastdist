// src/api/poisson.cpp
#include <fastdist/math/poisson.h>

extern "C" double fd_poisson_pmf(const double x, const double lambda) {
    return fastdist::math::poisson_pmf_scalar(x, lambda);
}

extern "C" double fd_poisson_cmf(const double x, const double lambda) {
    return fastdist::math::poisson_cdf_scalar(x, lambda);
}

extern "C" double fd_poisson_mean(const double lambda) { return fastdist::math::poisson_mean(lambda); }

extern "C" double fd_poisson_variance(const double lambda) { return fastdist::math::poisson_variance(lambda); }

extern "C" double fd_poisson_stddev(const double lambda) { return fastdist::math::poisson_stddev(lambda); }
