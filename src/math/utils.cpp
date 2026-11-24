// Statistical and combinatorial helpers. See include/fastdist/math/utils.h.
#include "fastdist/math/utils.h"
#include <cmath>
#include <limits>
#include <vector>

namespace fastdist::math {

    double chebyshev_bound(const double variance, const double k) {
        if (!std::isfinite(variance) || variance < 0.0 || !std::isfinite(k) || k <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return variance / (k * k);
    }

    double bayes_rule(const double p_B_given_A, const double p_A, const double p_B) {
        if (auto valid_prob = [](const double x) { return std::isfinite(x) && x >= 0.0 && x <= 1.0; };
            !valid_prob(p_B_given_A) || !valid_prob(p_A) || !valid_prob(p_B) || p_B == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (p_B_given_A * p_A) / p_B;
    }

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

    double law_of_total_probability(const std::vector<double>& probs_B_given_A, const std::vector<double>& probs_A) {
        if (probs_B_given_A.size() != probs_A.size() || probs_B_given_A.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return law_of_total_probability(probs_B_given_A.data(), probs_A.data(), probs_B_given_A.size());
    }

    double sigmoid(const double x) {
        if (!std::isfinite(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x >= 0.0) {
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            const double z = std::exp(x);
            return z / (1.0 + z);
        }
    }

    double logit(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p >= 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::log(p / (1.0 - p));
    }

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

    double coefficient_of_variation(const double mean, const double stddev) {
        if (!std::isfinite(mean) || !std::isfinite(stddev) || mean == 0.0 || stddev < 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return stddev / std::abs(mean);
    }

    double covariance(const double mean_x, const double mean_y, const double E_xy) {
        if (!std::isfinite(mean_x) || !std::isfinite(mean_y) || !std::isfinite(E_xy)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return E_xy - mean_x * mean_y;
    }

    double choose(const unsigned int n, unsigned int k) {
        if (k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;

        // Symmetry: C(n,k) == C(n,n-k), so loop over the smaller one.
        if (k > n - k) k = n - k;

        double result = 1.0;
        for (unsigned int i = 1; i <= k; ++i) {
            result *= (n - (k - i));
            result /= i;
        }

        return result;
    }

    double permutation(const unsigned int n, const unsigned int k) {
        if (k > n) return 0.0;
        if (k == 0) return 1.0;

        double result = 1.0;
        for (unsigned int i = 0; i < k; ++i) {
            result *= (n - i);
            // k can be in the billions; stop rather than multiply past infinity.
            if (std::isinf(result)) break;
        }

        return result;
    }

    double factorial(unsigned int n) {
        if (n == 0 || n == 1) return 1.0;

        // 171! is the first factorial a double cannot hold.
        constexpr unsigned int MAX_EXACT_FACTORIAL = 170;
        if (n > MAX_EXACT_FACTORIAL) return std::numeric_limits<double>::infinity();

        double result = 1.0;
        for (unsigned int i = 2; i <= n; ++i) {
            result *= i;
        }

        return result;
    }

    double gamma(const double x) { return std::tgamma(x); }

    double log_gamma(const double x) { return std::lgamma(x); }

    double binomial(const unsigned int n, const double a, const double b) {
        double result = 0.0;

        for (unsigned int k = 0; k <= n; ++k) {
            result += choose(n, k) * std::pow(a, n - k) * std::pow(b, k);
        }

        return result;
    }

    void sigmoid_cpu(const double* x_data, double* output, const size_t n) {
        for (size_t i = 0; i < n; i++) {
            output[i] = sigmoid(x_data[i]);
        }
    }
    void logit_cpu(const double* p_data, double* output, const size_t n) {
        for (size_t i = 0; i < n; i++) {
            output[i] = logit(p_data[i]);
        }
    }

} // namespace fastdist::math
