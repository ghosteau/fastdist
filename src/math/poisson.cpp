// Function declarations for poisson distribution functions
#include <fastdist/math/poisson.h>
#include <cmath>
#include <limits>

namespace fastdist::math {

    double poisson_pmf_scalar(double k, double lambda) {
        if (!std::isfinite(k) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Poisson is defined on non-negative integers
        if (k < 0.0 || std::floor(k) != k) {
            return 0.0;
        }

        // log PMF for numerical stability:
        // log P = k * log(lambda) - lambda - log(k!)
        const double log_p =
            k * std::log(lambda) - lambda - std::lgamma(k + 1.0);

        return std::exp(log_p);
    }

    double poisson_cdf_scalar(double k, double lambda) {
        if (!std::isfinite(k) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < 0.0) {
            return 0.0;
        }

        const int ki = static_cast<int>(std::floor(k));

        double sum = 0.0;
        for (int i = 0; i <= ki; ++i) {
            sum += poisson_pmf_scalar(i, lambda);
        }

        return sum;
    }

    double poisson_mean(double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return lambda;
    }

    double poisson_variance(double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return lambda;
    }

    double poisson_stddev(double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(lambda);
    }

} // namespace fastdist::math
