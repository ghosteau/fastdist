// Function declarations for continuous uniform distribution functions
#include <cmath>
#include <fastdist/math/uniform.h>
#include <limits>
#include <random>

// Note that the default uniform distribution is continuous for this implementation
namespace fastdist::math {

    double uniform_pdf_scalar(const double x, const double a, const double b) {
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

    double uniform_cdf_scalar(const double x, const double a, const double b) {
        // Check parameters
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x <= a) return 0.0;
        if (x >= b) return 1.0;

        return (x - a) / (b - a);
    }

    double uniform_mean(const double a, const double b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return 0.5 * (a + b);
    }

    double uniform_variance(const double a, const double b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (b - a) * (b - a) / 12.0;
    }

    double uniform_stddev(const double a, const double b) {
        // Basic validity check
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::sqrt((b - a) * (b - a) / 12.0);
    }

    // MGF: M_X(t) = (exp(b t) - exp(a t)) / (t * (b - a)), t != 0; M_X(0) = 1
    double uniform_mgf_scalar(const double t, const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(t)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (t == 0.0) return 1.0;

        return (std::exp(b * t) - std::exp(a * t)) / (t * (b - a));
    }

    // CGF: K_X(t) = log(M_X(t))
    double uniform_cgf_scalar(const double t, const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(t)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double mgf = uniform_mgf_scalar(t, a, b);
        if (mgf <= 0.0) return std::numeric_limits<double>::quiet_NaN();

        return std::log(mgf);
    }

    // RNG: simple thread-local uniform_real_distribution
    double uniform_sample(const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution dist(a, b);
        return dist(rng);
    }

    // Batch Functions
    void uniform_pdf_batch(const double* x_data, double* output, const size_t n, const double a, const double b,
                           const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = uniform_pdf_scalar(x_data[i] + stepSize * static_cast<double>(i), a, b);
        }
    }
    void uniform_cdf_batch(const double* x_data, double* output, const size_t n, const double a, const double b,
                           const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = uniform_cdf_scalar(x_data[i] + stepSize * static_cast<double>(i), a, b);
        }
    }

    void uniform_mgf_batch(const double* t_data, double* output, const size_t n, const double a, const double b,
                           const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = uniform_mgf_scalar(t_data[i] + stepSize * static_cast<double>(i), a, b);
        }
    }

    void uniform_cgf_batch(const double* t_data, double* output, const size_t n, const double a, const double b,
                           const double stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = uniform_cgf_scalar(t_data[i] + stepSize * static_cast<double>(i), a, b);
        }
    }

} // namespace fastdist::math
