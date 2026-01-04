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

extern "C" double fd_poisson_mgf(const double t, const double lambda) {
    return fastdist::math::poisson_mgf_scalar(t, lambda);
}

extern "C" double fd_poisson_cgf(const double t, const double lambda) {
    return fastdist::math::poisson_cgf_scalar(t, lambda);
}

extern "C" int fd_poisson_sample(const double lambda) { return fastdist::math::poisson_sample(lambda); }
