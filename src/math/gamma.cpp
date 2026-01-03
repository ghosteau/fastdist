// Function declarations for gamma distribution functions
#include <cmath>
#include <config.h>
#include <fastdist/math/gamma.h>
#include <limits>
#include <random>

namespace fastdist::math {

    // -------------------------
    // PDF
    // f(x) = x^{α-1} e^{-x/θ} / (Γ(α) θ^α)
    // -------------------------
    double gamma_pdf_scalar(const double x, const double alpha, const double theta) {
        if (!std::isfinite(x) || !std::isfinite(alpha) || !std::isfinite(theta) || x < 0.0 || alpha <= 0.0 ||
            theta <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::pow(x, alpha - 1.0) * std::exp(-x / theta) / (std::tgamma(alpha) * std::pow(theta, alpha));
    }

    // Forward declarations for internal functions
    static double gamma_p_series(double a, double x);
    static double gamma_p_cf(double a, double x);

    // -------------------------
    // CDF using series / continued fraction
    // -------------------------
    double gamma_cdf_scalar(const double x, const double alpha, const double theta) {
        if (!std::isfinite(x) || !std::isfinite(alpha) || !std::isfinite(theta) || x < 0.0 || alpha <= 0.0 ||
            theta <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double z = x / theta;
        if (z < alpha + 1.0) {
            return gamma_p_series(alpha, z);
        } else {
            return gamma_p_cf(alpha, z);
        }
    }

    // -------------------------
    // Mean, variance, stddev
    // -------------------------
    double gamma_mean(const double alpha, const double theta) {
        if (!std::isfinite(alpha) || !std::isfinite(theta) || alpha <= 0.0 || theta <= 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        return alpha * theta;
    }

    double gamma_variance(const double alpha, const double theta) {
        if (!std::isfinite(alpha) || !std::isfinite(theta) || alpha <= 0.0 || theta <= 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        return alpha * theta * theta;
    }

    double gamma_stddev(const double alpha, const double theta) { return std::sqrt(gamma_variance(alpha, theta)); }

    // -------------------------
    // MGF and CGF
    // M_X(t) = (1 - θ t)^{-α}, t < 1/θ
    // -------------------------
    double gamma_mgf_scalar(const double t, const double alpha, const double theta) {
        if (!std::isfinite(t) || !std::isfinite(alpha) || !std::isfinite(theta) || alpha <= 0.0 || theta <= 0.0 ||
            t >= 1.0 / theta) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::pow(1.0 - theta * t, -alpha);
    }

    double gamma_cgf_scalar(const double t, const double alpha, const double theta) {
        double mgf = gamma_mgf_scalar(t, alpha, theta);
        if (!std::isfinite(mgf)) return std::numeric_limits<double>::quiet_NaN();
        return std::log(mgf);
    }

    // -------------------------
    // RNG
    // -------------------------
    double gamma_sample(const double alpha, const double theta) {
        if (!std::isfinite(alpha) || !std::isfinite(theta) || alpha <= 0.0 || theta <= 0.0)
            return std::numeric_limits<double>::quiet_NaN();

        thread_local std::mt19937 rng{std::random_device{}()};
        std::gamma_distribution dist(alpha, theta);
        return dist(rng);
    }

    // -------------------------
    // Internal: lower incomplete gamma series representation
    // -------------------------
    static double gamma_p_series(const double a, const double x) {
        double sum = 1.0 / a;
        double term = sum;

        for (unsigned int n = 1; n <= MAX_ITER; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::fabs(term) < EPS * std::fabs(sum)) break;
        }

        return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }

    // -------------------------
    // Internal functions: continued fraction representation via Lentz's method
    // -------------------------
    static double gamma_p_cf(const double a, const double x) {
        double b = x + 1.0 - a;
        double c = 1.0 / FPMIN;
        double d = 1.0 / b;
        double h = d;

        for (unsigned int i = 1; i <= MAX_ITER; ++i) {
            const double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (std::fabs(d) < FPMIN) d = FPMIN;
            c = b + an / c;
            if (std::fabs(c) < FPMIN) c = FPMIN;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (std::fabs(delta - 1.0) < EPS) break;
        }

        return 1.0 - std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
    }

} // namespace fastdist::math
