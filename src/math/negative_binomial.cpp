// Function declarations for negative binomial distribution functions
#include <algorithm>
#include <cmath>
#include <fastdist/math/negative_binomial.h>
#include <limits>
#include <random>

namespace fastdist::math {

    // -------------------------
    // PMF: P(X=k) = C(k+r-1, k) * p^r * (1-p)^k
    // k = number of failures, r = number of successes, p = success probability
    // -------------------------
    double negative_binomial_pmf_scalar(const int k, const int r, const double p) {
        // Parameter validation
        if (!std::isfinite(p) || p <= 0.0 || p >= 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Outside support
        if (k < 0) {
            return 0.0;
        }

        const double comb = std::tgamma(k + r) / (std::tgamma(r) * std::tgamma(k + 1.0));

        return comb * std::pow(p, r) * std::pow(1.0 - p, k);
    }

    // -------------------------
    // CDF: sum_{i=0}^{k} PMF(i)
    // -------------------------
    double negative_binomial_cdf_scalar(const int k, const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p >= 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < 0) {
            return 0.0;
        }

        double sum = 0.0;
        for (int i = 0; i <= k; ++i) {
            sum += negative_binomial_pmf_scalar(i, r, p);
        }
        // Summing PMF terms accumulates rounding error, so the total can
        // land a few ULP above 1.0
        return std::min(sum, 1.0);
    }


    // -------------------------
    // Mean: r * (1-p)/p
    // -------------------------
    double negative_binomial_mean(const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return r * (1.0 - p) / p;
    }

    // -------------------------
    // Variance: r * (1-p) / p^2
    // -------------------------
    double negative_binomial_variance(const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return r * (1.0 - p) / (p * p);
    }

    // -------------------------
    // Standard deviation
    // -------------------------
    double negative_binomial_stddev(const int r, const double p) {
        double var = negative_binomial_variance(r, p);
        return std::sqrt(var);
    }

    // -------------------------
    // MGF: M(t) = (p / (1 - (1-p)e^t))^r
    // -------------------------
    double negative_binomial_mgf_scalar(const double t, const int r, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p <= 0.0 || p >= 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double denom = 1.0 - (1.0 - p) * std::exp(t);
        if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();

        return std::pow(p / denom, r);
    }

    // -------------------------
    // CGF: log MGF
    // -------------------------
    double negative_binomial_cgf_scalar(const double t, const int r, const double p) {
        double mgf = negative_binomial_mgf_scalar(t, r, p);
        if (!std::isfinite(mgf)) return std::numeric_limits<double>::quiet_NaN();
        return std::log(mgf);
    }

    // -------------------------
    // Random sample using standard library
    // -------------------------
    int negative_binomial_sample(const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p >= 1.0 || r <= 0) {
            return -1; // invalid input
        }

        thread_local std::mt19937 rng{std::random_device{}()};
        std::negative_binomial_distribution dist(r, p);
        return dist(rng);
    }

} // namespace fastdist::math
