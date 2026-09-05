// Function definitions for discrete uniform distribution functions
#include <cmath>
#include <fastdist/math/discrete_uniform.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

namespace fastdist::math {

    double discrete_uniform_pmf_scalar(const int x, const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < a || x > b) {
            return 0.0;
        }

        const auto n = static_cast<double>(b - a + 1);
        return 1.0 / n;
    }

    double discrete_uniform_cdf_scalar(const int x, const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < a) return 0.0;
        if (x >= b) return 1.0;

        const auto n = static_cast<double>(b - a + 1);
        return static_cast<double>(x - a + 1) / n;
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

        const auto n = static_cast<double>(b - a + 1);
        return (n * n - 1.0) / 12.0;
    }

    double discrete_uniform_stddev(const int a, const int b) {
        if (a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto n = static_cast<double>(b - a + 1);
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

        const auto n = static_cast<double>(b - a + 1);
        const double et = std::exp(t);

        return std::exp(t * a) * (std::pow(et, n) - 1.0) / (n * (et - 1.0));
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
