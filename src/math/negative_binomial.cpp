// Function definitions for negative binomial distribution functions
#include <algorithm>
#include <cmath>
#include <fastdist/math/negative_binomial.h>
#include <fastdist/math/rng.h>
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

        // Evaluated in log space. Forming C(k + r - 1, k) from raw factorials
        // overflows a double once k + r - 1 > 170 -- inf at k = 170 and nan
        // beyond -- even though the coefficient and the resulting PMF are
        // comfortably inside range (for r = 3, k = 200 the true PMF is 1.6e-57).
        // lgamma keeps the intermediate values small, and folding the two pows
        // into the same exponent removes them from the hot path.
        const double log_pmf = std::lgamma(static_cast<double>(k) + r) - std::lgamma(static_cast<double>(r)) -
                               std::lgamma(static_cast<double>(k) + 1.0) + r * std::log(p) +
                               static_cast<double>(k) * std::log1p(-p);

        return std::exp(log_pmf);
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

        // Consecutive PMF terms satisfy
        //     P(i) = P(i-1) * ((i + r - 1) / i) * (1 - p)
        // so the sum costs one exp overall instead of three tgammas and two
        // pows per term.
        //
        // P(0) = p^r underflows for small p with large r, which would collapse
        // the recurrence to zero; fall back to per-term evaluation there.
        const double log_p0 = static_cast<double>(r) * std::log(p);
        constexpr double MIN_RECURRENCE_LOG = -700.0;

        if (log_p0 > MIN_RECURRENCE_LOG) {
            const double q = 1.0 - p;
            double term = std::exp(log_p0);
            double sum = term;
            for (int i = 1; i <= k; ++i) {
                term *= (static_cast<double>(i + r - 1) / static_cast<double>(i)) * q;
                sum += term;
            }
            // Summing PMF terms accumulates rounding error, so the total can
            // land a few ULP above 1.0
            return std::min(sum, 1.0);
        }

        double sum = 0.0;
        for (int i = 0; i <= k; ++i) {
            sum += negative_binomial_pmf_scalar(i, r, p);
        }
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

        std::negative_binomial_distribution dist(r, p);
        return dist(rng());
    }

} // namespace fastdist::math
