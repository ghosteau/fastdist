// Function declarations for exponential distribution functions
#include "fastdist/math/exponential.h"
#include <cmath>
#include <limits>
#include <random>

namespace fastdist::math {

    double exponential_pdf_scalar(const double x, const double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x < 0.0) {
            return 0.0;
        }
        return lambda * std::exp(-lambda * x);
    }

    double exponential_cdf_scalar(const double x, const double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x < 0.0) {
            return 0.0;
        }
        return 1.0 - std::exp(-lambda * x);
    }

    double exponential_mean(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / lambda;
    }

    double exponential_variance(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / (lambda * lambda);
    }

    double exponential_stddev(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / lambda;
    }

    // M_X(t) = lambda / (lambda - t), t < lambda
    double exponential_mgf_scalar(const double t, const double lambda) {
        if (!std::isfinite(t) || !std::isfinite(lambda) || lambda <= 0.0 || t >= lambda) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return lambda / (lambda - t);
    }

    double exponential_cgf_scalar(const double t, const double lambda) {
        if (!std::isfinite(t) || !std::isfinite(lambda) || lambda <= 0.0 || t >= lambda) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::log(lambda) - std::log(lambda - t);
    }

    // X ~ Exponential(lambda)
    double exponential_sample(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        thread_local std::mt19937 rng{std::random_device{}()};
        std::exponential_distribution dist(lambda);

        return dist(rng);
    }

} // namespace fastdist::math
