// Function declarations for geometric distribution functions
#include <cmath>
#include <fastdist/math/geometric.h>
#include <limits>

namespace fastdist::math {

    double geometric_pmf_scalar(const int k, const double p) {
        // p must be in (0,1]
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Geometric is defined for k >= 1
        if (k < 1) {
            return 0.0;
        }

        return p * std::pow(1.0 - p, k - 1);
    }

    double geometric_cdf_scalar(const int k, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < 1) {
            return 0.0;
        }

        return 1.0 - std::pow(1.0 - p, k);
    }

    double geometric_mean(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / p;
    }

    double geometric_variance(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (1.0 - p) / (p * p);
    }

    double geometric_stddev(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt((1.0 - p) / (p * p));
    }

    // TODO: Add MGF

} // namespace fastdist::math
