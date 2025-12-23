// Function definitions for extra library math utility functions
#include "fastdist/math/utils.h"
#include <cmath>
#include <limits>

namespace fastdist::math {

    // -------------------------
    // Chebyshev-Bienaymé bound
    // -------------------------
    double chebyshev_bound(const double variance, const double k) {
        if (!std::isfinite(variance) || variance < 0.0 || !std::isfinite(k) || k <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return variance / (k * k);
    }

    // -------------------------
    // Bayes' Theorem
    // -------------------------
    double bayes_rule(const double p_B_given_A, const double p_A, const double p_B) {
        auto valid_prob = [](double x) { return std::isfinite(x) && x >= 0.0 && x <= 1.0; };

        if (!valid_prob(p_B_given_A) || !valid_prob(p_A) || !valid_prob(p_B) || p_B == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (p_B_given_A * p_A) / p_B;
    }

    // -------------------------
    // Sigmoid function
    // σ(x) = 1 / (1 + e^-x)
    // -------------------------
    double sigmoid(const double x) {
        if (!std::isfinite(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Numerically stable sigmoid
        if (x >= 0.0) {
            double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            double z = std::exp(x);
            return z / (1.0 + z);
        }
    }

    // -------------------------
    // Logit function
    // logit(p) = log(p / (1 - p))
    // -------------------------
    double logit(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p >= 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::log(p / (1.0 - p));
    }

    // -------------------------
    // Euclidean distance (1D)
    // -------------------------
    double euclidean_distance(const double x, const double y) {
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::abs(x - y);
    }

    // -------------------------
    // Manhattan distance (1D)
    // -------------------------
    double manhattan_distance(const double x, const double y) {
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::abs(x - y);
    }

    // -------------------------
    // Coefficient of Variation
    // CV = stddev / mean
    // -------------------------
    double coefficient_of_variation(const double mean, const double stddev) {
        if (!std::isfinite(mean) || !std::isfinite(stddev) || mean == 0.0 || stddev < 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return stddev / std::abs(mean);
    }

    // -------------------------
    // Covariance (population)
    // Cov(X,Y) = E[(X - μx)(Y - μy)]
    // Cov(X,Y) = E[XY] - E[X]E[Y]
    // -------------------------
    double covariance(const double mean_x, const double mean_y, const double E_xy) {
        if (!std::isfinite(mean_x) || !std::isfinite(mean_y) || !std::isfinite(E_xy)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return E_xy - mean_x * mean_y;
    }

} // namespace fastdist::math
