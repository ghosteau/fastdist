// Function definitions for Bernoulli distribution functions
#include <cmath>
#include <fastdist/math/bernoulli.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

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

    // M_X(t) = E[e^{tX}] = (1 - p) + p e^t
    double bernoulli_mgf_scalar(const double t, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (1.0 - p) + p * std::exp(t);
    }

    double bernoulli_cgf_scalar(const double t, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::log((1.0 - p) + p * std::exp(t));
    }

    // X ~ Bernoulli(p)
    int bernoulli_sample(const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return -1; // signal invalid input
        }

        std::bernoulli_distribution dist(p);

        return dist(rng()) ? 1 : 0;
    }

    // Batch Functions
    void bernoulli_pmf_batch(const int* k_data, double* output, const size_t n, const double p, const int stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = bernoulli_pmf_scalar(k_data[i] + stepSize * static_cast<int>(i), p);
        }
    }

    void bernoulli_cdf_batch(const int* k_data, double* output, const size_t n, const double p, const int stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = bernoulli_cdf_scalar(k_data[i] + stepSize * static_cast<int>(i), p);
        }
    }

    void bernoulli_mgf_batch(const double* t_data, double* output, const size_t n, const double p, const int stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = bernoulli_mgf_scalar(t_data[i] + stepSize * static_cast<int>(i), p);
        }
    }

    void bernoulli_cgf_batch(const double* t_data, double* output, const size_t n, const double p, const int stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = bernoulli_cgf_scalar(t_data[i] + stepSize * static_cast<int>(i), p);
        }
    }

} // namespace fastdist::math
