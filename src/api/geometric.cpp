// src/api/geometric.cpp
#include <fastdist/math/geometric.h>

extern "C" double fd_geometric_pmf(int k, double p) { return fastdist::math::geometric_pmf_scalar(k, p); }

extern "C" double fd_geometric_cmf(int k, double p) { return fastdist::math::geometric_cdf_scalar(k, p); }

extern "C" double fd_geometric_mean(double p) { return fastdist::math::geometric_mean(p); }

extern "C" double fd_geometric_variance(double p) { return fastdist::math::geometric_variance(p); }

extern "C" double fd_geometric_stddev(double p) { return fastdist::math::geometric_stddev(p); }
