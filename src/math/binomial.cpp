// Function definitions for binomial distribution functions
#include <algorithm>
#include <cmath>
#include <fastdist/math/binomial.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

namespace fastdist::math {

    // Computes log PMF
    double binomial_logpmf_scalar(const int x, const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < 0 || x > n) return -std::numeric_limits<double>::infinity();
        if (p == 0.0) return (x == 0 ? 0.0 : -std::numeric_limits<double>::infinity());
        if (p == 1.0) return (x == n ? 0.0 : -std::numeric_limits<double>::infinity());

        const double log_coeff = std::lgamma(n + 1.0) - std::lgamma(x + 1.0) - std::lgamma(n - x + 1.0);
        return log_coeff + x * std::log(p) + (n - x) * std::log1p(-p);
    }

    // PMF uses log PMF for efficiency
    double binomial_pmf_scalar(const int x, const int n, const double p) {
        return std::exp(binomial_logpmf_scalar(x, n, p));
    }

    // CDF sums PMF for k = 0..x
    double binomial_cdf_scalar(const int x, const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < 0) return 0.0;
        if (x >= n) return 1.0;

        // p == 1 puts all mass at n, and x < n here, so nothing has accumulated
        // yet. Handled separately because the ratio below divides by (1 - p).
        if (p == 1.0) return 0.0;

        // Consecutive PMF terms satisfy
        //     P(k) = P(k-1) * ((n - k + 1) / k) * (p / (1 - p))
        // so the sum costs one exp overall instead of three lgammas, two logs
        // and an exp per term.
        //
        // P(0) = (1-p)^n underflows for large n, which would collapse the whole
        // recurrence to zero; fall back to per-term log-space evaluation there.
        const double log_p0 = static_cast<double>(n) * std::log1p(-p);
        constexpr double MIN_RECURRENCE_LOG = -700.0;

        if (log_p0 > MIN_RECURRENCE_LOG) {
            const double odds = p / (1.0 - p);
            double term = std::exp(log_p0);
            double sum = term;
            for (int k = 1; k <= x; ++k) {
                term *= (static_cast<double>(n - k + 1) / static_cast<double>(k)) * odds;
                sum += term;
            }
            // Summing PMF terms accumulates rounding error, so the total can
            // land a few ULP above 1.0
            return std::min(sum, 1.0);
        }

        double sum = 0.0;
        for (int k = 0; k <= x; ++k) {
            sum += binomial_pmf_scalar(k, n, p);
        }
        return std::min(sum, 1.0);
    }

    double binomial_mean(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return n * p;
    }

    double binomial_variance(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return n * p * (1.0 - p);
    }

    double binomial_stddev(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(n * p * (1.0 - p));
    }

    // M_X(t) = ( (1 - p) + p e^t )^n
    double binomial_mgf_scalar(const double t, const int n, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::pow((1.0 - p) + p * std::exp(t), n);
    }

    // K_X(t) = log M_X(t)
    double binomial_cgf_scalar(const double t, const int n, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return n * std::log((1.0 - p) + p * std::exp(t));
    }

    // X ~ Binomial(n, p)
    int binomial_sample(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return -1; // signal invalid input
        }

        std::binomial_distribution<int> dist(n, p);

        return dist(rng());
    }

} // namespace fastdist::math
