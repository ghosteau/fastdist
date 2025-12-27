// src/api/bernoulli.cpp
#include <fastdist/math/bernoulli.h>

extern "C" double fd_bernoulli_pmf(const int k, const double p) { return fastdist::math::bernoulli_pmf_scalar(k, p); }

extern "C" double fd_bernoulli_cdf(const int k, const double p) { return fastdist::math::bernoulli_cdf_scalar(k, p); }

extern "C" double fd_bernoulli_mean(const double p) { return fastdist::math::bernoulli_mean(p); }

extern "C" double fd_bernoulli_variance(const double p) { return fastdist::math::bernoulli_variance(p); }

extern "C" double fd_bernoulli_stddev(const double p) { return fastdist::math::bernoulli_stddev(p); }

extern "C" double fd_bernoulli_mgf(const double t, const double p) {
    return fastdist::math::bernoulli_mgf_scalar(t, p);
}

extern "C" double fd_bernoulli_cgf(const double t, const double p) {
    return fastdist::math::bernoulli_cgf_scalar(t, p);
}

extern "C" int fd_bernoulli_sample(const double p) { return fastdist::math::bernoulli_sample(p); }
