// Function definitions for normal distribution functions
#include <algorithm>
#include <cmath>
#include <fastdist/math/normal.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <math/constants.h>
#include <random>

namespace fastdist::math {

    namespace {
        // Formula cores shared by the scalar and batch paths. Every term that
        // depends only on the parameters is an argument, so a batch call computes
        // it once per array. Both paths order the arithmetic identically, so they
        // agree bit for bit.
        inline double normal_pdf_core(const double x, const double mu, const double sigma, const double denom) {
            const double z = (x - mu) / sigma;
            return std::exp(-0.5 * z * z) / denom;
        }

        inline double normal_logpdf_core(const double x, const double mu, const double inv_sigma,
                                         const double log_sigma) {
            const double z = (x - mu) * inv_sigma;
            return -0.5 * z * z - log_sigma - LOG_SQRT_2PI;
        }

        inline double normal_cdf_core(const double x, const double mu, const double scale) {
            return 0.5 * (1.0 + std::erf((x - mu) / scale));
        }
    } // namespace

    double normal_pdf_scalar(const double x, const double mu, const double sigma) {
        if (!std::isfinite(x) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return normal_pdf_core(x, mu, sigma, sigma * SQRT_2PI);
    }

    double normal_logpdf_scalar(const double x, const double mu, const double sigma) {
        if (!std::isfinite(x) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return normal_logpdf_core(x, mu, 1.0 / sigma, std::log(sigma));
    }

    double normal_cdf_scalar(const double x, const double mu, const double sigma) {
        if (!std::isfinite(x) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return normal_cdf_core(x, mu, sigma * std::sqrt(2.0));
    }

    double normal_mean(const double mu) {
        if (!std::isfinite(mu)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return mu;
    }

    double normal_variance(const double sigma) {
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return sigma * sigma;
    }

    double normal_stddev(const double sigma) {
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return sigma;
    }

    double normal_mgf_scalar(const double t, const double mu, const double sigma) {
        if (!std::isfinite(t) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::exp(mu * t + 0.5 * sigma * sigma * t * t);
    }

    double normal_cgf_scalar(const double t, const double mu, const double sigma) {
        if (!std::isfinite(t) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return mu * t + 0.5 * sigma * sigma * t * t;
    }

    double normal_sample(const double mu, const double sigma) {
        if (!std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::normal_distribution<double> dist(mu, sigma);
        return dist(rng());
    }

    double normal_log_sample(const double mu, const double sigma) {
        if (!std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::normal_distribution dist(mu, sigma);
        return std::exp(dist(rng()));
    }

    double z_score(const double x, const double mu, const double sigma) { return (x - mu) / sigma; }

    // Batch functions evaluate at x_data[i] + stepSize * i. Invalid parameters
    // make every output NaN; a non-finite input makes only its own output NaN.
    void normal_pdf_batch(const double* x_data, double* output, const size_t n, const double mu, const double sigma,
                          const double stepSize) {
        if (!std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        const double denom = sigma * SQRT_2PI;

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            output[i] = normal_pdf_core(x, mu, sigma, denom);
        }
    }

    void normal_logpdf_batch(const double* x_data, double* output, const size_t n, const double mu, const double sigma,
                             const double stepSize) {
        if (!std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        const double inv_sigma = 1.0 / sigma;
        const double log_sigma = std::log(sigma);

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            output[i] = normal_logpdf_core(x, mu, inv_sigma, log_sigma);
        }
    }

    void normal_cdf_batch(const double* x_data, double* output, const size_t n, const double mu, const double sigma,
                          const double stepSize) {
        if (!std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        const double scale = sigma * std::sqrt(2.0);

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            output[i] = normal_cdf_core(x, mu, scale);
        }
    }

    void normal_mgf_batch(const double* t_data, double* output, const size_t n, const double mu, const double sigma,
                          const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = normal_mgf_scalar(t_data[i] + stepSize * static_cast<double>(i), mu, sigma);
        }
    }

    void normal_cgf_batch(const double* t_data, double* output, const size_t n, const double mu, const double sigma,
                          const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = normal_cgf_scalar(t_data[i] + stepSize * static_cast<double>(i), mu, sigma);
        }
    }

} // namespace fastdist::math
