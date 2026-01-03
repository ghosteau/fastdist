// Function declarations for normal distribution functions
#include <cmath>
#include <fastdist/math/normal.h>
#include <limits>

namespace fastdist::math {

    double normal_pdf_scalar(const double x, const double mu, const double sigma) {
        if (!std::isfinite(x) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double z = (x - mu) / sigma;
        return std::exp(-0.5 * z * z) / (sigma * SQRT_2PI);
    }

    double normal_logpdf_scalar(const double x, const double mu, const double sigma) {
        if (!std::isfinite(x) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double inv_sigma = 1.0 / sigma;
        const double z = (x - mu) * inv_sigma;
        return -0.5 * z * z - std::log(sigma) - LOG_SQRT_2PI;
    }

    double normal_cdf_scalar(const double x, const double mu, const double sigma) {
        if (!std::isfinite(x) || !std::isfinite(mu) || !std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double z = (x - mu) / (sigma * std::sqrt(2.0));
        return 0.5 * (1.0 + std::erf(z));
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

    double z_score(const double x, const double mu, const double sigma) { return (x - mu) / sigma; }

    // Batch Functions
    void normal_pdf_batch(const double* x_data, double* output, const size_t n, const double mu, const double sigma) {
        for (size_t i = 0; i < n; i++) {
            output[i] = normal_pdf_scalar(x_data[i], mu, sigma);
        }
    }

} // namespace fastdist::math
