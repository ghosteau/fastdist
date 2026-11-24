// Continuous uniform distribution. See include/probforge/math/uniform.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <probforge/math/rng.h>
#include <probforge/math/uniform.h>
#include <random>

namespace probforge::math {

    double uniform_pdf_scalar(const double x, const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < a || x > b) {
            return 0.0;
        }

        return 1.0 / (b - a);
    }

    double uniform_cdf_scalar(const double x, const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x <= a) return 0.0;
        if (x >= b) return 1.0;

        return (x - a) / (b - a);
    }

    double uniform_mean(const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return 0.5 * (a + b);
    }

    double uniform_variance(const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (b - a) * (b - a) / 12.0;
    }

    double uniform_stddev(const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::sqrt((b - a) * (b - a) / 12.0);
    }

    double uniform_mgf_scalar(const double t, const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(t)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (t == 0.0) return 1.0;

        // expm1 on both terms: the -1 cancels in the difference, but
        // exp(bt) - exp(at) loses the leading digits for small t. At t = 1e-13
        // the plain form left only four correct digits.
        return (std::expm1(b * t) - std::expm1(a * t)) / (t * (b - a));
    }

    double uniform_cgf_scalar(const double t, const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b || !std::isfinite(t)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double mgf = uniform_mgf_scalar(t, a, b);
        if (mgf <= 0.0) return std::numeric_limits<double>::quiet_NaN();

        return std::log(mgf);
    }

    double uniform_sample(const double a, const double b) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::uniform_real_distribution dist(a, b);
        return dist(rng());
    }

    // Invalid parameters make every output NaN; a non-finite input only its own.
    void uniform_pdf_batch(const double* x_data, double* output, const size_t n, const double a, const double b,
                           const double stepSize) {
        // The density is constant on [a, b], so hoist it.
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        const double density = 1.0 / (b - a);

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            output[i] = (x < a || x > b) ? 0.0 : density;
        }
    }
    void uniform_cdf_batch(const double* x_data, double* output, const size_t n, const double a, const double b,
                           const double stepSize) {
        if (!std::isfinite(a) || !std::isfinite(b) || a >= b) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        // A division, not a reciprocal multiply, to stay bit-identical to the scalar path.
        const double range = b - a;

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            if (x <= a) {
                output[i] = 0.0;
            } else if (x >= b) {
                output[i] = 1.0;
            } else {
                output[i] = (x - a) / range;
            }
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

} // namespace probforge::math
