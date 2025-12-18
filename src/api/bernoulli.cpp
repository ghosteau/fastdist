// src/api/bernoulli.cpp
#include <fastdist/math/bernoulli.h>

extern "C" double fd_bernoulli_pmf(int k, double p) {
    return fastdist::math::bernoulli_pmf_scalar(k, p);
}

extern "C" double fd_bernoulli_cdf(int k, double p) {
    return fastdist::math::bernoulli_cdf_scalar(k, p);
}

extern "C" double fd_bernoulli_mean(double p) {
    return fastdist::math::bernoulli_mean(p);
}

extern "C" double fd_bernoulli_variance(double p) {
    return fastdist::math::bernoulli_variance(p);
}

extern "C" double fd_bernoulli_stddev(double p) {
    return fastdist::math::bernoulli_stddev(p);
}
