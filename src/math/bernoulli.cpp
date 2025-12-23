// Function declarations for Bernoulli distribution functions
#include <cmath>
#include <fastdist/math/bernoulli.h>
#include <limits>

namespace fastdist::math {

    double bernoulli_pmf_scalar(const int k, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Bernoulli is only defined for k = 0 or 1
        if (k != 0 && k != 1) {
            return 0.0;
        }

        return (k == 1) ? p : (1.0 - p);
    }

    double bernoulli_cdf_scalar(const int k, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < 0) return 0.0;
        if (k < 1) return 1.0 - p;
        return 1.0;
    }

    double bernoulli_mean(const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return p;
    }

    double bernoulli_variance(const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return p * (1.0 - p);
    }

    double bernoulli_stddev(const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(p * (1.0 - p));
    }

    // TODO: Add MGF

} // namespace fastdist::math
