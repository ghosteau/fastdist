// Discrete uniform distribution. See include/probforge/math/discrete_uniform.h.
#include <cmath>
#include <limits>
#include <probforge/math/discrete_uniform.h>
#include <probforge/math/rng.h>
#include <random>

namespace probforge::math {

    namespace {
        // Widened before the subtraction: b - a + 1 in int wraps for a range
        // wider than INT_MAX, which made the PMF infinite and the variance negative.
        inline double support_size(const int a, const int b) {
            return static_cast<double>(b) - static_cast<double>(a) + 1.0;
        }
    } // namespace

    double discrete_uniform_pmf_scalar(const int x, const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < a || x > b) {
            return 0.0;
        }

        return 1.0 / support_size(a, b);
    }

    double discrete_uniform_cdf_scalar(const int x, const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < a) return 0.0;
        if (x >= b) return 1.0;

        const double rank = static_cast<double>(x) - static_cast<double>(a) + 1.0;
        return rank / support_size(a, b);
    }

    double discrete_uniform_mean(const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 0.5 * (static_cast<double>(a) + static_cast<double>(b));
    }

    double discrete_uniform_variance(const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double n = support_size(a, b);
        return (n * n - 1.0) / 12.0;
    }

    double discrete_uniform_stddev(const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double n = support_size(a, b);
        return std::sqrt((n * n - 1.0) / 12.0);
    }

    double discrete_uniform_mgf_scalar(const double t, const int a, const int b) {
        if (!std::isfinite(t) || a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (t == 0.0) {
            return 1.0;
        }

        const double n = support_size(a, b);

        // expm1 on both: written as (e^t)^n - 1 over e^t - 1, both subtractions
        // cancel for small t, and the power squares its rounding error on top.
        return std::exp(t * static_cast<double>(a)) * std::expm1(t * n) / (n * std::expm1(t));
    }

    double discrete_uniform_cgf_scalar(const double t, const int a, const int b) {
        const double mgf = discrete_uniform_mgf_scalar(t, a, b);
        if (!std::isfinite(mgf) || mgf <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::log(mgf);
    }

    int discrete_uniform_sample(const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<int>::min();
        }

        std::uniform_int_distribution<int> dist(a, b);

        return dist(rng());
    }

} // namespace probforge::math
