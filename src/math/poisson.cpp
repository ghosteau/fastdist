// Poisson distribution. See include/probforge/math/poisson.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <probforge/math/poisson.h>
#include <probforge/math/rng.h>
#include <random>

namespace probforge::math {
    double poisson_pmf_scalar(const double x, const double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Defined on the non-negative integers only.
        if (x < 0.0 || std::floor(x) != x) {
            return 0.0;
        }

        // Log space: lambda^k / k! overflows both terms for modest k even where
        // their ratio is an ordinary number.
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

        // Stays a double: x need only be finite, so it can exceed INT_MAX, where
        // the conversion is undefined. It wrapped to INT_MIN and skipped the loop.
        const double k = std::floor(x);

        // P(i) = P(i-1) * lambda / i, so the sum costs one exp. It starts from
        // P(0) = exp(-lambda), which underflows for large lambda and would collapse
        // the whole sum; fall back to per-term log-space evaluation there.
        constexpr double MAX_RECURRENCE_LAMBDA = 700.0;

        if (lambda <= MAX_RECURRENCE_LAMBDA) {
            double term = std::exp(-lambda);
            double sum = term;
            for (double i = 1.0; i <= k; i += 1.0) {
                term *= lambda / i;
                const double next = sum + term;
                // Terms only shrink past the mode at lambda, so one that no longer
                // moves the sum there means every later term is negligible too.
                if (i > lambda && next == sum) break;
                sum = next;
            }
            // Accumulated rounding can leave the total a few ULP above 1.
            return std::min(sum, 1.0);
        }

        const double log_lambda = std::log(lambda);
        double sum = 0.0;
        for (double i = 0.0; i <= k; i += 1.0) {
            const double term = std::exp(i * log_lambda - lambda - std::lgamma(i + 1.0));
            const double next = sum + term;
            if (i > lambda && next == sum) break;
            sum = next;
        }
        return std::min(sum, 1.0);
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

    double poisson_mgf_scalar(const double t, const double lambda) {
        if (!std::isfinite(t) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::exp(lambda * (std::exp(t) - 1.0));
    }

    double poisson_cgf_scalar(const double t, const double lambda) {
        if (!std::isfinite(t) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return lambda * (std::exp(t) - 1.0);
    }

    int poisson_sample(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return -1; // signal invalid input
        }
        std::poisson_distribution<int> dist(lambda);
        return dist(rng());
    }

    // Invalid parameters make every output NaN; a non-finite input only its own.
    void poisson_pmf_batch(const double* x_data, double* output, const size_t n, const double lambda,
                           const int stepSize) {
        // lambda is fixed across the array, so validate and take its log once.
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
            return;
        }

        const double log_lambda = std::log(lambda);

        for (size_t i = 0; i < n; i++) {
            const double x = x_data[i] + stepSize * static_cast<double>(i);
            if (!std::isfinite(x)) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            if (x < 0.0 || std::floor(x) != x) {
                output[i] = 0.0;
                continue;
            }
            output[i] = std::exp(x * log_lambda - lambda - std::lgamma(x + 1.0));
        }
    }
    void poisson_cdf_batch(const double* x_data, double* output, const size_t n, const double lambda,
                           const int stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = poisson_cdf_scalar(x_data[i] + stepSize * static_cast<double>(i), lambda);
        }
    }

    void poisson_mgf_batch(const double* t_data, double* output, const size_t n, const double lambda,
                           const int stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = poisson_mgf_scalar(t_data[i] + stepSize * static_cast<double>(i), lambda);
        }
    }

    void poisson_cgf_batch(const double* t_data, double* output, const size_t n, const double lambda,
                           const int stepSize) {
        for (size_t i = 0; i < n; i++) {
            output[i] = poisson_cgf_scalar(t_data[i] + stepSize * static_cast<double>(i), lambda);
        }
    }

} // namespace probforge::math
