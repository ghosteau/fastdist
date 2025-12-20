// src/api/discrete_uniform.cpp
#include <fastdist/math/discrete_uniform.h>

extern "C" double fd_discrete_uniform_pmf(int x, int a, int b) {
    return fastdist::math::discrete_uniform_pmf_scalar(x, a, b);
}

extern "C" double fd_discrete_uniform_cmf(int x, int a, int b) {
    return fastdist::math::discrete_uniform_cdf_scalar(x, a, b);
}

extern "C" double fd_discrete_uniform_mean(int a, int b) { return fastdist::math::discrete_uniform_mean(a, b); }

extern "C" double fd_discrete_uniform_variance(int a, int b) { return fastdist::math::discrete_uniform_variance(a, b); }

extern "C" double fd_discrete_uniform_stddev(int a, int b) { return fastdist::math::discrete_uniform_stddev(a, b); }
