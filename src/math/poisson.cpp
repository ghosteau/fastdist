// Function declarations for poisson distribution functions
#include <cmath>
#include <fastdist/math/poisson.h>
#include <limits>
#include <random>

namespace fastdist::math {
    double poisson_pmf_scalar(const double x, const double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Poisson is defined
        // on non-negative
        // integers
        if (x < 0.0 || std::floor(x) != x) {
            return 0.0;
        }

        // log PMF for
        // numerical
        // stability: log P =
        // k * log(lambda) -
        // lambda - log(k!)
        const double log_p = x * std::log(lambda) - lambda - std::lgamma(x + 1.0);

        return std::exp(log_p);
    }

    double poisson_cdf_scalar(const double x, const double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < 0.0) {
            return 0.0;
        }

        const int ki = static_cast<int>(std::floor(x));

        double sum = 0.0;
        for (int i = 0; i <= ki; ++i) {
            sum += poisson_pmf_scalar(i, lambda);
        }

        return sum;
    }

    double poisson_mean(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return lambda;
    }

    double poisson_variance(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return lambda;
    }

    double poisson_stddev(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(lambda);
    }

    // MGF: M_X(t) = exp(lambda * (exp(t) - 1))
    double poisson_mgf_scalar(const double t, const double lambda) {
        if (!std::isfinite(t) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::exp(lambda * (std::exp(t) - 1.0));
    }

    // CGF: K_X(t) = lambda * (exp(t) - 1)
    double poisson_cgf_scalar(const double t, const double lambda) {
        if (!std::isfinite(t) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return lambda * (std::exp(t) - 1.0);
    }

    // RNG via std::poisson_distribution
    int poisson_sample(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return -1; // signal invalid input
        }
        thread_local std::mt19937 rng{std::random_device{}()};
        std::poisson_distribution<int> dist(lambda);
        return dist(rng);
    }

} // namespace fastdist::math
