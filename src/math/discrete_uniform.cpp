// Function declarations for discrete uniform distribution functions
#include <cmath>
#include <fastdist/math/discrete_uniform.h>
#include <limits>

namespace fastdist::math {

    double discrete_uniform_pmf_scalar(const int x, const int a, const int b) {
        // Check parameters: a <= b, finite integers
        if (!std::isfinite(a) || !std::isfinite(b) || a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // x must be integer in [a, b]
        if (x < a || x > b) {
            return 0.0;
        }

        const double n = static_cast<double>(b - a + 1);
        return 1.0 / n;
    }

    double discrete_uniform_cdf_scalar(const int x, const int a, const int b) {
        // Check parameters
        if (!std::isfinite(a) || !std::isfinite(b) || a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Below support
        if (x < a) {
            return 0.0;
        }

        // Above or equal upper support
        if (x >= b) {
            return 1.0;
        }

        // Middle region: (x - a + 1)/(b - a + 1)
        const auto n = static_cast<double>(b - a + 1);
        const auto count = static_cast<double>(x - a + 1);
        return count / n;
    }

    double discrete_uniform_mean(int a, int b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 0.5 * (static_cast<double>(a) + static_cast<double>(b));
    }

    double discrete_uniform_variance(int a, int b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto n = static_cast<double>(b - a + 1);
        // Var = ((n^2) - 1) / 12
        return (n * n - 1.0) / 12.0;
    }

    double discrete_uniform_stddev(int a, int b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a > b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto n = static_cast<double>(b - a + 1);
        return std::sqrt((n * n - 1.0) / 12.0);
    }

} // namespace fastdist::math
