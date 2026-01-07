// Function declarations for binomial distribution functions
#include <cmath>
#include <fastdist/math/binomial.h>
#include <limits>
#include <random>

namespace fastdist::math {

    // Computes log PMF
    double binomial_logpmf_scalar(const int x, const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < 0 || x > n) return -std::numeric_limits<double>::infinity();
        if (p == 0.0) return (x == 0 ? 0.0 : -std::numeric_limits<double>::infinity());
        if (p == 1.0) return (x == n ? 0.0 : -std::numeric_limits<double>::infinity());

        const double log_coeff = std::lgamma(n + 1.0) - std::lgamma(x + 1.0) - std::lgamma(n - x + 1.0);
        return log_coeff + x * std::log(p) + (n - x) * std::log1p(-p);
    }

    // PMF uses log PMF for efficiency
    double binomial_pmf_scalar(const int x, const int n, const double p) {
        return std::exp(binomial_logpmf_scalar(x, n, p));
    }

    // CDF sums PMF for k = 0..x
    double binomial_cdf_scalar(const int x, const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x < 0) return 0.0;
        if (x >= n) return 1.0;

        double sum = 0.0;
        for (int k = 0; k <= x; ++k) {
            sum += binomial_pmf_scalar(k, n, p);
        }
        return sum;
    }

    double binomial_mean(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return n * p;
    }

    double binomial_variance(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return n * p * (1.0 - p);
    }

    double binomial_stddev(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(n * p * (1.0 - p));
    }

    // M_X(t) = ( (1 - p) + p e^t )^n
    double binomial_mgf_scalar(const double t, const int n, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::pow((1.0 - p) + p * std::exp(t), n);
    }

    // K_X(t) = log M_X(t)
    double binomial_cgf_scalar(const double t, const int n, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return n * std::log((1.0 - p) + p * std::exp(t));
    }

    // X ~ Binomial(n, p)
    int binomial_sample(const int n, const double p) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0 || n < 0) {
            return -1; // signal invalid input
        }

        thread_local std::mt19937 rng{std::random_device{}()};
        std::binomial_distribution<int> dist(n, p);

        return dist(rng);
    }

} // namespace fastdist::math
