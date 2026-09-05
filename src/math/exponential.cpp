// Function declarations for exponential distribution functions
#include "fastdist/math/exponential.h"
#include <algorithm>
#include <cmath>
#include <fastdist/math/rng.h>
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

        std::exponential_distribution dist(lambda);

        return dist(rng());
    }

    // Batch Functions
    void exponential_pdf_batch(const double* x_data, double* output, const size_t n, const double lambda,
                               const double stepSize) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            output[i] = (x < 0.0) ? 0.0 : lambda * std::exp(-lambda * x);
        }
    }

    void exponential_cdf_batch(const double* x_data, double* output, const size_t n, const double lambda,
                               const double stepSize) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            output[i] = (x < 0.0) ? 0.0 : 1.0 - std::exp(-lambda * x);
        }
    }

    void exponential_mgf_batch(const double* t_data, double* output, const size_t n, const double lambda,
                               const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = exponential_mgf_scalar(t_data[i] + stepSize * static_cast<double>(i), lambda);
        }
    }

    void exponential_cgf_batch(const double* t_data, double* output, const size_t n, const double lambda,
                               const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = exponential_cgf_scalar(t_data[i] + stepSize * static_cast<double>(i), lambda);
        }
    }

} // namespace fastdist::math
