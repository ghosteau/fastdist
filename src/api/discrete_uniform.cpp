// src/api/discrete_uniform.cpp
#include <fastdist/math/discrete_uniform.h>

extern "C" double fd_discrete_uniform_pmf(const int x, const int a, const int b) {
    return fastdist::math::discrete_uniform_pmf_scalar(x, a, b);
}

extern "C" double fd_discrete_uniform_cmf(const int x, const int a, const int b) {
    return fastdist::math::discrete_uniform_cdf_scalar(x, a, b);
}

extern "C" double fd_discrete_uniform_mean(const int a, const int b) {
    return fastdist::math::discrete_uniform_mean(a, b);
}

extern "C" double fd_discrete_uniform_variance(const int a, const int b) {
    return fastdist::math::discrete_uniform_variance(a, b);
}

extern "C" double fd_discrete_uniform_stddev(const int a, const int b) {
    return fastdist::math::discrete_uniform_stddev(a, b);
}
