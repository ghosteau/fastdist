// Function declarations for exponential distribution functions
#include "fastdist/math/exponential.h"
#include <cmath>
#include <limits>

namespace fastdist::math {

    double exponential_pdf_scalar(double x, double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x < 0.0) {
            return 0.0;
        }
        return lambda * std::exp(-lambda * x);
    }

    double exponential_cdf_scalar(double x, double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x < 0.0) {
            return 0.0;
        }
        return 1.0 - std::exp(-lambda * x);
    }

    double exponential_mean(double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / lambda;
    }

    double exponential_variance(double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / (lambda * lambda);
    }

    double exponential_stddev(double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / lambda;
    }

} // namespace fastdist::math
