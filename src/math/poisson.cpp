// Function definitions for poisson distribution functions
#include <algorithm>
#include <cmath>
#include <fastdist/math/poisson.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

namespace fastdist::math {
    double poisson_pmf_scalar(const double x, const double lambda) {
        if (!std::isfinite(x) || !std::isfinite(lambda) || lambda <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Poisson is defined on non-negative integers
        if (x < 0.0 || std::floor(x) != x) {
            return 0.0;
        }

        // Evaluated in log space for numerical stability:
        //     log P(k) = k log(lambda) - lambda - log(k!)
        // Forming lambda^k / k! directly overflows both terms for modest k
        // even where their ratio is an ordinary number.
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

        // Consecutive PMF terms are related by P(i) = P(i-1) * lambda / i, so
        // the sum needs one exp in total rather than a log, an lgamma and an
        // exp per term. That is the difference between this being the slowest
        // path in the library and it being competitive -- see BENCHMARKS.md.
        //
        // The recurrence has to start from P(0) = exp(-lambda), which underflows
        // to zero for large lambda and would collapse the whole sum to zero even
        // where the true CDF is O(1). Past that point, fall back to evaluating
        // each term in log space, which stays accurate because the exponent
        // i*log(lambda) - lambda - lgamma(i+1) remains small near i = lambda.
        constexpr double MAX_RECURRENCE_LAMBDA = 700.0;

        if (lambda <= MAX_RECURRENCE_LAMBDA) {
            double term = std::exp(-lambda);
            double sum = term;
            for (int i = 1; i <= ki; ++i) {
                term *= lambda / static_cast<double>(i);
                sum += term;
            }
            // Summing PMF terms accumulates rounding error, so the total can
            // land a few ULP above 1.0
            return std::min(sum, 1.0);
        }

        const double log_lambda = std::log(lambda);
        double sum = 0.0;
        for (int i = 0; i <= ki; ++i) {
            sum += std::exp(static_cast<double>(i) * log_lambda - lambda - std::lgamma(i + 1.0));
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

    // X ~ Poisson(lambda)
    int poisson_sample(const double lambda) {
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            return -1; // signal invalid input
        }
        std::poisson_distribution<int> dist(lambda);
        return dist(rng());
    }

    // Batch Functions
    void poisson_pmf_batch(const double* x_data, double* output, const size_t n, const double lambda,
                           const int stepSize) {
        // lambda is fixed across the array, so both its validation and log() are
        // hoisted; log(lambda) used to be a transcendental call per element for a
        // value that never changes.
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

} // namespace fastdist::math
