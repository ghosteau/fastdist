// src/api/geometric.cpp
#include <fastdist/math/geometric.h>

extern "C" double fd_geometric_pmf(const int k, const double p) { return fastdist::math::geometric_pmf_scalar(k, p); }

extern "C" double fd_geometric_cmf(const int k, const double p) { return fastdist::math::geometric_cdf_scalar(k, p); }

extern "C" double fd_geometric_mean(const double p) { return fastdist::math::geometric_mean(p); }

extern "C" double fd_geometric_variance(const double p) { return fastdist::math::geometric_variance(p); }

extern "C" double fd_geometric_stddev(const double p) { return fastdist::math::geometric_stddev(p); }

extern "C" double fd_geometric_mgf(const double t, const double p) {
    return fastdist::math::geometric_mgf_scalar(t, p);
}

extern "C" double fd_geometric_cgf(const double t, const double p) {
    return fastdist::math::geometric_cgf_scalar(t, p);
}

extern "C" int fd_geometric_sample(const double p) { return fastdist::math::geometric_sample(p); }
