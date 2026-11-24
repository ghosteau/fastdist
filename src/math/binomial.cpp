// Binomial distribution. See include/probforge/math/binomial.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <probforge/math/binomial.h>
#include <probforge/math/rng.h>
#include <random>

namespace probforge::math {

    // Log space: C(n, x) overflows from factorials past n = 170.
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

    double binomial_pmf_scalar(const int x, const int n, const double p) {
        return std::exp(binomial_logpmf_scalar(x, n, p));
    }

    double binomial_cdf_scalar(const int x, const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < 0) return 0.0;
        if (x >= n) return 1.0;

        // All mass sits at n and x < n here. Also avoids dividing by (1 - p) below.
        if (p == 1.0) return 0.0;

        // P(k) = P(k-1) * ((n-k+1) / k) * (p / (1-p)), so the sum costs one exp.
        // P(0) = (1-p)^n underflows for large n, which would collapse the whole
        // recurrence; fall back to per-term log-space evaluation there.
        const double log_p0 = static_cast<double>(n) * std::log1p(-p);
        constexpr double MIN_RECURRENCE_LOG = -700.0;

        // Terms only shrink past the mode, so one that no longer moves the sum
        // there means every later term is negligible too. n can be in the billions.
        const double mode = (static_cast<double>(n) + 1.0) * p;

        if (log_p0 > MIN_RECURRENCE_LOG) {
            const double odds = p / (1.0 - p);
            double term = std::exp(log_p0);
            double sum = term;
            for (int k = 1; k <= x; ++k) {
                term *= (static_cast<double>(n - k + 1) / static_cast<double>(k)) * odds;
                const double next = sum + term;
                if (static_cast<double>(k) > mode && next == sum) break;
                sum = next;
            }
            // Accumulated rounding can leave the total a few ULP above 1.
            return std::min(sum, 1.0);
        }

        double sum = 0.0;
        for (int k = 0; k <= x; ++k) {
            const double next = sum + binomial_pmf_scalar(k, n, p);
            if (static_cast<double>(k) > mode && next == sum) break;
            sum = next;
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

    double binomial_mgf_scalar(const double t, const int n, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::pow((1.0 - p) + p * std::exp(t), n);
    }

    double binomial_cgf_scalar(const double t, const int n, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return n * std::log((1.0 - p) + p * std::exp(t));
    }

    int binomial_sample(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return -1; // signal invalid input
        }

        std::binomial_distribution<int> dist(n, p);

        return dist(rng());
    }

} // namespace probforge::math
