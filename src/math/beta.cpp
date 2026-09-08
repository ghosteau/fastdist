// Function definitions for beta distribution functions
#include <cmath>
#include <config.h>
#include <fastdist/math/beta.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

namespace fastdist::math {

    // -------------------------
    // PDF
    // f(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
    // -------------------------
    double beta_pdf_scalar(const double x, const double alpha, const double beta) {
        // Parameter validation
        if (!std::isfinite(x) || !std::isfinite(alpha) || !std::isfinite(beta) || alpha <= 0.0 || beta <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Outside support
        if (x < 0.0 || x > 1.0) {
            return 0.0;
        }

        const double B = std::tgamma(alpha) * std::tgamma(beta) / std::tgamma(alpha + beta);

        return std::pow(x, alpha - 1.0) * std::pow(1.0 - x, beta - 1.0) / B;
    }

    // Forward declarations for internal functions
    static double beta_continued_fraction(double a, double b, double x);

    // -------------------------
    // CDF
    //
    // The regularized incomplete beta I_x(a,b), evaluated as
    //
    //     I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * CF(a,b,x)
    //
    // where CF is the standard continued fraction for the function. The
    // fraction converges rapidly for x below the transition point
    // (a+1)/(a+b+2) and slowly above it, so the reflection
    //
    //     I_x(a,b) = 1 - I_{1-x}(b,a)
    //
    // maps every input onto the fast side before evaluating.
    //
    // The leading factor is formed in log space. Computing B(a,b) directly
    // overflows for large a or b, and x^a underflows for large a, even when
    // their combination is an ordinary number.
    // -------------------------
    double beta_cdf_scalar(const double x, const double alpha, const double beta) {
        if (!std::isfinite(x) || !std::isfinite(alpha) || !std::isfinite(beta) || alpha <= 0.0 || beta <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;

        const double log_prefactor = std::lgamma(alpha + beta) - std::lgamma(alpha) - std::lgamma(beta) +
                                     alpha * std::log(x) + beta * std::log1p(-x);
        const double prefactor = std::exp(log_prefactor);

        double result;
        if (x < (alpha + 1.0) / (alpha + beta + 2.0)) {
            result = prefactor * beta_continued_fraction(alpha, beta, x) / alpha;
        } else {
            result = 1.0 - prefactor * beta_continued_fraction(beta, alpha, 1.0 - x) / beta;
        }

        // The reflection above subtracts two nearby quantities in the upper
        // tail, so the result can land a few ULP outside [0, 1]. A CDF that
        // reports 1 + 1e-16 breaks callers that treat it as a probability.
        return std::min(std::max(result, 0.0), 1.0);
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
        // Every other sampler validates its parameters; this one did not, and
        // std::gamma_distribution has undefined behaviour for a non-positive
        // shape rather than a defined error value.
        if (!std::isfinite(alpha) || !std::isfinite(beta) || alpha <= 0.0 || beta <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::gamma_distribution<double> ga(alpha, 1.0);
        std::gamma_distribution<double> gb(beta, 1.0);
        double a = ga(rng());
        double b = gb(rng());
        return a / (a + b);
    }

    // -------------------------
    // Internal: incomplete beta series
    // -------------------------
    // Modified Lentz evaluation of the continued fraction for the incomplete
    // beta function (Numerical Recipes 6.4). Each iteration applies two
    // coefficients, the even and odd terms of the fraction.
    //
    // FPMIN guards the standard Lentz failure mode: a denominator that lands
    // exactly on zero would otherwise propagate an infinity through the whole
    // recurrence.
    static double beta_continued_fraction(const double a, const double b, const double x) {
        const double qab = a + b;
        const double qap = a + 1.0;
        const double qam = a - 1.0;

        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (std::fabs(d) < FPMIN) d = FPMIN;
        d = 1.0 / d;
        double h = d;

        for (unsigned int m = 1; m <= MAX_ITER; ++m) {
            const double m2 = 2.0 * m;

            // Even step.
            double numerator = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + numerator * d;
            if (std::fabs(d) < FPMIN) d = FPMIN;
            c = 1.0 + numerator / c;
            if (std::fabs(c) < FPMIN) c = FPMIN;
            d = 1.0 / d;
            h *= d * c;

            // Odd step.
            numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + numerator * d;
            if (std::fabs(d) < FPMIN) d = FPMIN;
            c = 1.0 + numerator / c;
            if (std::fabs(c) < FPMIN) c = FPMIN;
            d = 1.0 / d;
            const double delta = d * c;
            h *= delta;

            if (std::fabs(delta - 1.0) < EPS) break;
        }

        return h;
    }

} // namespace fastdist::math
