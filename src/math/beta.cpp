// Function declarations for beta distribution functions
#include <cmath>
#include <config.h>
#include <fastdist/math/beta.h>
#include <limits>
#include <random>

namespace fastdist::math {

    // -------------------------
    // PDF
    // f(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
    // -------------------------
    double beta_pdf_scalar(const double x, const double alpha, const double beta) {
        if (!std::isfinite(x) || !std::isfinite(alpha) || !std::isfinite(beta) || x < 0.0 || x > 1.0 || alpha <= 0.0 ||
            beta <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double B = std::tgamma(alpha) * std::tgamma(beta) / std::tgamma(alpha + beta);
        return std::pow(x, alpha - 1.0) * std::pow(1.0 - x, beta - 1.0) / B;
    }

    // Forward declarations for internal functions
    static double beta_inc_series(double a, double b, double x);

    // -------------------------
    // CDF using series
    // -------------------------
    double beta_cdf_scalar(const double x, const double alpha, const double beta) {
        if (!std::isfinite(x) || !std::isfinite(alpha) || !std::isfinite(beta) || x < 0.0 || x > 1.0 || alpha <= 0.0 ||
            beta <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Use symmetry for better convergence
        if (x < (alpha + 1.0) / (alpha + beta + 2.0)) {
            return beta_inc_series(alpha, beta, x);
        } else {
            return 1.0 - beta_inc_series(beta, alpha, 1.0 - x);
        }
    }

    // -------------------------
    // Mean, variance, stddev
    // -------------------------
    double beta_mean(const double alpha, const double beta) { return alpha / (alpha + beta); }

    double beta_variance(const double alpha, const double beta) {
        return (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0));
    }

    double beta_stddev(const double alpha, const double beta) { return std::sqrt(beta_variance(alpha, beta)); }

    // -------------------------
    // RNG
    // -------------------------
    double beta_sample(const double alpha, const double beta) {
        thread_local std::mt19937 rng{std::random_device{}()};
        std::gamma_distribution<double> ga(alpha, 1.0);
        std::gamma_distribution<double> gb(beta, 1.0);
        double a = ga(rng);
        double b = gb(rng);
        return a / (a + b);
    }

    // -------------------------
    // Internal: incomplete beta series
    // -------------------------
    static double beta_inc_series(const double a, const double b, const double x) {
        double sum = 1.0 / a;
        double term = sum;
        for (unsigned int n = 1; n <= MAX_ITER; ++n) {
            term *= x * (a + n - 1) / (a + b + n - 1);
            sum += term;
            if (std::fabs(term) < EPS * std::fabs(sum)) break;
        }
        return sum * std::pow(x, a) * std::pow(1.0 - x, b) / std::tgamma(a + 1.0);
    }

} // namespace fastdist::math
