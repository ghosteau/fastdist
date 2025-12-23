// src/api/utils.cpp
#include "fastdist/math/utils.h"

extern "C" double fd_chebyshev_bound(const double variance, const double k) {
    return fastdist::math::chebyshev_bound(variance, k);
}

extern "C" double fd_bayes_rule(const double p_B_given_A, const double p_A, const double p_B) {
    return fastdist::math::bayes_rule(p_B_given_A, p_A, p_B);
}

extern "C" double fd_sigmoid(const double x) { return fastdist::math::sigmoid(x); }

extern "C" double fd_logit(const double p) { return fastdist::math::logit(p); }

extern "C" double fd_euclidean_distance(const double x, const double y) {
    return fastdist::math::euclidean_distance(x, y);
}

extern "C" double fd_manhattan_distance(const double x, const double y) {
    return fastdist::math::manhattan_distance(x, y);
}

extern "C" double fd_coefficient_of_variation(const double mean, const double stddev) {
    return fastdist::math::coefficient_of_variation(mean, stddev);
}

extern "C" double fd_covariance(const double mean_x, const double mean_y, const double E_xy) {
    return fastdist::math::covariance(mean_x, mean_y, E_xy);
}
