// Negative binomial distribution. See include/probforge/math/negative_binomial.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <probforge/math/negative_binomial.h>
#include <probforge/math/rng.h>
#include <random>

namespace probforge::math {

    double negative_binomial_pmf_scalar(const int k, const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Outside support
        if (k < 0) {
            return 0.0;
        }

        // Special-cased because the log form below evaluates 0 * log(0) here.
        if (p == 1.0) {
            return (k == 0) ? 1.0 : 0.0;
        }

        // Log space: C(k + r - 1, k) overflows from factorials past 170.
        const double log_pmf = std::lgamma(static_cast<double>(k) + r) - std::lgamma(static_cast<double>(r)) -
                               std::lgamma(static_cast<double>(k) + 1.0) + r * std::log(p) +
                               static_cast<double>(k) * std::log1p(-p);

        return std::exp(log_pmf);
    }

    double negative_binomial_cdf_scalar(const int k, const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < 0) {
            return 0.0;
        }

        // P(i) = P(i-1) * ((i + r - 1) / i) * (1 - p), so the sum costs one exp.
        // P(0) = p^r underflows for small p with large r, which would collapse the
        // recurrence to zero; fall back to per-term log-space evaluation there.
        const double log_p0 = static_cast<double>(r) * std::log(p);
        constexpr double MIN_RECURRENCE_LOG = -700.0;

        // Terms only shrink past the mode, so one that no longer moves the sum
        // there means every later term is negligible too. k is unbounded above.
        const double mode = (static_cast<double>(r) - 1.0) * (1.0 - p) / p;

        if (log_p0 > MIN_RECURRENCE_LOG) {
            const double q = 1.0 - p;
            double term = std::exp(log_p0);
            double sum = term;
            for (int i = 1; i <= k; ++i) {
                term *= (static_cast<double>(i + r - 1) / static_cast<double>(i)) * q;
                const double next = sum + term;
                if (static_cast<double>(i) > mode && next == sum) break;
                sum = next;
            }
            // Accumulated rounding can leave the total a few ULP above 1.
            return std::min(sum, 1.0);
        }

        double sum = 0.0;
        for (int i = 0; i <= k; ++i) {
            const double next = sum + negative_binomial_pmf_scalar(i, r, p);
            if (static_cast<double>(i) > mode && next == sum) break;
            sum = next;
        }
        return std::min(sum, 1.0);
    }


    double negative_binomial_mean(const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return r * (1.0 - p) / p;
    }

    double negative_binomial_variance(const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return r * (1.0 - p) / (p * p);
    }

    double negative_binomial_stddev(const int r, const double p) {
        double var = negative_binomial_variance(r, p);
        return std::sqrt(var);
    }

    double negative_binomial_mgf_scalar(const double t, const int r, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double denom = 1.0 - (1.0 - p) * std::exp(t);
        if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();

        return std::pow(p / denom, r);
    }

    // From the formula, not log(MGF): the MGF overflows for large r.
    double negative_binomial_cgf_scalar(const double t, const int r, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // log1p keeps the subtraction accurate when (1-p)e^t is small.
        const double scaled = (1.0 - p) * std::exp(t);
        if (scaled >= 1.0) return std::numeric_limits<double>::quiet_NaN();

        return r * (std::log(p) - std::log1p(-scaled));
    }

    int negative_binomial_sample(const int r, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0 || r <= 0) {
            return -1; // invalid input
        }

        // std::negative_binomial_distribution is undefined at p == 1.
        if (p == 1.0) return 0;

        std::negative_binomial_distribution dist(r, p);
        return dist(rng());
    }

} // namespace probforge::math
