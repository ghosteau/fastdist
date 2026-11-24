// Function definitions for discrete uniform distribution functions
#include <cmath>
#include <fastdist/math/discrete_uniform.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

namespace fastdist::math {

    namespace {
        // Support size, widened before the subtraction. b - a + 1 in int
        // overflows whenever the range spans more than INT_MAX values: for
        // a = INT_MIN, b = INT_MAX it wrapped to 0, which made the PMF
        // infinite and the variance negative.
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

    // M_X(t) = e^{ta} (e^{t(b-a+1)} - 1) / ((b-a+1)(e^t - 1))
    double discrete_uniform_mgf_scalar(const double t, const int a, const int b) {
        if (!std::isfinite(t) || a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (t == 0.0) {
            return 1.0;
        }

        const double n = support_size(a, b);

        // expm1 on both the numerator and the denominator. Written as
        // (e^t)^n - 1 over e^t - 1 the two subtractions each cancel for small
        // t -- at t = 1e-10 that left only eight correct digits -- and raising
        // e^t to the n-th power squares its rounding error on top.
        return std::exp(t * static_cast<double>(a)) * std::expm1(t * n) / (n * std::expm1(t));
    }

    double discrete_uniform_cgf_scalar(const double t, const int a, const int b) {
        const double mgf = discrete_uniform_mgf_scalar(t, a, b);
        if (!std::isfinite(mgf) || mgf <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::log(mgf);
    }

    // X ~ DiscreteUniform(a, b)
    int discrete_uniform_sample(const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<int>::min();
        }

        std::uniform_int_distribution<int> dist(a, b);

        return dist(rng());
    }

} // namespace fastdist::math
