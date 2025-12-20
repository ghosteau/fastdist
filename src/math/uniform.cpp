// Function declarations for continuous uniform distribution functions
#include <cmath>
#include <fastdist/math/uniform.h>
#include <limits>

// Note that the default uniform distribution is continuous for this implementation
namespace fastdist::math {

    double uniform_pdf_scalar(double x, double a, double b) {
        // Check parameters: a < b, finite numbers
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // PDF is zero outside [a, b]
        if (x < a || x > b) {
            return 0.0;
        }

        return 1.0 / (b - a);
    }

    double uniform_cdf_scalar(double x, double a, double b) {
        // Check parameters
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x <= a) return 0.0;
        if (x >= b) return 1.0;

        return (x - a) / (b - a);
    }

    double uniform_mean(double a, double b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return 0.5 * (a + b);
    }

    double uniform_variance(double a, double b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (b - a) * (b - a) / 12.0;
    }

    double uniform_stddev(double a, double b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::sqrt((b - a) * (b - a) / 12.0);
    }

} // namespace fastdist::math
