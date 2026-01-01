// src/api/utils.cpp
#include "fastdist/math/utils.h"

extern "C" double fd_chebyshev_bound(const double variance, const double k) {
    return fastdist::math::chebyshev_bound(variance, k);
}

extern "C" double fd_bayes_rule(const double p_B_given_A, const double p_A, const double p_B) {
    return fastdist::math::bayes_rule(p_B_given_A, p_A, p_B);
}

// Pointer overload for law of total probability
extern "C" double fd_law_of_total_probability(const double* probs_B_given_A, const double* probs_A, const size_t n) {
    return fastdist::math::law_of_total_probability(probs_B_given_A, probs_A, n);
}

// Vector overload for law of total probability
extern "C" double fd_law_of_total_probability_vec(const double* probs_B_given_A, const double* probs_A,
                                                  const size_t n) {
    // Wrap raw arrays into vectors to call the safe overload
    std::vector vec_B(probs_B_given_A, probs_B_given_A + n);
    std::vector vec_A(probs_A, probs_A + n);
    return fastdist::math::law_of_total_probability(vec_B, vec_A);
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

extern "C" double fd_choose(const unsigned int n, const unsigned int k) { return fastdist::math::choose(n, k); }

extern "C" double fd_permutation(const unsigned int n, const unsigned int k) {
    return fastdist::math::permutation(n, k);
}

extern "C" double fd_factorial(const unsigned int n) { return fastdist::math::factorial(n); }

extern "C" double fd_gamma(const double x) { return fastdist::math::gamma(x); }

extern "C" double fd_log_gamma(const double x) { return fastdist::math::log_gamma(x); }

extern "C" double fd_binomial(const unsigned int n, const double a, const double b) {
    return fastdist::math::binomial(n, a, b);
}
