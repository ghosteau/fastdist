// Function definitions for extra library math utility functions
#include "fastdist/math/utils.h"
#include <cmath>
#include <limits>
#include <vector>

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
        if (auto valid_prob = [](const double x) { return std::isfinite(x) && x >= 0.0 && x <= 1.0; };
            !valid_prob(p_B_given_A) || !valid_prob(p_A) || !valid_prob(p_B) || p_B == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (p_B_given_A * p_A) / p_B;
    }

    // -------------------------
    // Law of Total Probability
    // P(B) = Σ P(B|A_i) * P(A_i)
    // -------------------------
    double law_of_total_probability(const double* probs_B_given_A, const double* probs_A, size_t n) {
        if (!probs_B_given_A || !probs_A || n == 0) return std::numeric_limits<double>::quiet_NaN();

        double total = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double pB_given_A = probs_B_given_A[i];
            double pA = probs_A[i];
            if (!(std::isfinite(pB_given_A) && std::isfinite(pA) && pB_given_A >= 0.0 && pB_given_A <= 1.0 &&
                  pA >= 0.0 && pA <= 1.0)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            total += pB_given_A * pA;
        }
        return total;
    }

    // -------------------------
    // Law of Total Probability (vector wrapper)
    // P(B) = Σ P(B|A_i) * P(A_i)
    // Note: This is overloaded to make Python usage easier and safe
    // -------------------------
    double law_of_total_probability(const std::vector<double>& probs_B_given_A, const std::vector<double>& probs_A) {
        if (probs_B_given_A.size() != probs_A.size() || probs_B_given_A.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return law_of_total_probability(probs_B_given_A.data(), probs_A.data(), probs_B_given_A.size());
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
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            const double z = std::exp(x);
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
    // Euclidean distance (n-D)
    // sqrt(sum_i (x_i - y_i)^2)
    // -------------------------
    double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double sum_sq = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double diff = x[i] - y[i];
            sum_sq += diff * diff;
        }

        return std::sqrt(sum_sq);
    }

    // -------------------------
    // Manhattan distance (n-D)
    // sum_i |x_i - y_i|
    // -------------------------
    double manhattan_distance(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double sum_abs = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            sum_abs += std::abs(x[i] - y[i]);
        }

        return sum_abs;
    }

    // -------------------------
    // Cosine similarity (n-D)
    // cos_sim(x,y) = (x·y) / (||x|| * ||y||)
    // -------------------------
    double cosine_similarity(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double dot = 0.0;
        double norm_x = 0.0;
        double norm_y = 0.0;

        for (size_t i = 0; i < x.size(); ++i) {
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            dot += x[i] * y[i];
            norm_x += x[i] * x[i];
            norm_y += y[i] * y[i];
        }

        if (norm_x == 0.0 || norm_y == 0.0) return std::numeric_limits<double>::quiet_NaN();

        return dot / (std::sqrt(norm_x) * std::sqrt(norm_y));
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

    // -------------------------
    // Binomial Coefficient / Combinatorial Function
    // C(n,k) = n! / (k!(n-k)!)
    // -------------------------
    double choose(const unsigned int n, unsigned int k) {
        if (k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;

        // symmetry relationship
        if (k > n - k) k = n - k;

        double result = 1.0;
        for (unsigned int i = 1; i <= k; ++i) {
            result *= (n - (k - i));
            result /= i;
        }

        return result;
    }

    // -------------------------
    // Permutations
    // P(n,k) = n! / (n-k)!
    // -------------------------
    double permutation(const unsigned int n, const unsigned int k) {
        if (k > n) return 0.0;
        if (k == 0) return 1.0;

        double result = 1.0;
        for (unsigned int i = 0; i < k; ++i) {
            result *= (n - i);
        }

        return result;
    }

    // -------------------------
    // Factorial
    // n! = n * (n-1) * (n-2) * ... * 1
    // -------------------------
    double factorial(unsigned int n) {
        if (n == 0 || n == 1) return 1.0;

        double result = 1.0;
        for (unsigned int i = 2; i <= n; ++i) {
            result *= i;
        }

        return result;
    }

    // -------------------------
    // Gamma Function
    // Γ(x) = integral(t^(x-1) e^(-t) dt) from zero to infinity
    // Γ(n+1) = n! for integer n
    // -------------------------
    double gamma(const double x) { return std::tgamma(x); }

    // -------------------------
    // Log Gamma Function
    // log Γ(x)
    // Numerically stable alternative to log(gamma(x))
    // -------------------------
    double log_gamma(const double x) { return std::lgamma(x); }

    // -------------------------
    // Binomial Theorem / Binomial Expansion
    // (a + b)^n = sum(C(n,k) a^(n-k)b^k)
    // -------------------------
    double binomial(const unsigned int n, const double a, const double b) {
        double result = 0.0;

        for (unsigned int k = 0; k <= n; ++k) {
            result += choose(n, k) * std::pow(a, n - k) * std::pow(b, k);
        }

        return result;
    }

} // namespace fastdist::math
