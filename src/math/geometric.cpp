// Geometric distribution. See include/probforge/math/geometric.h.
#include <cmath>
#include <limits>
#include <probforge/math/geometric.h>
#include <probforge/math/rng.h>
#include <random>

namespace probforge::math {

    double geometric_pmf_scalar(const int k, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Support starts at 1.
        if (k < 1) {
            return 0.0;
        }

        return p * std::pow(1.0 - p, k - 1);
    }

    double geometric_cdf_scalar(const int k, const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < 1) {
            return 0.0;
        }

        // 1 - (1-p)^k cancels for small p: at p = 1e-15 it cost three of four
        // correct digits. The expm1 log form keeps full precision throughout.
        return -std::expm1(static_cast<double>(k) * std::log1p(-p));
    }

    double geometric_mean(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / p;
    }

    double geometric_variance(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (1.0 - p) / (p * p);
    }

    double geometric_stddev(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt((1.0 - p) / (p * p));
    }

    double geometric_mgf_scalar(const double t, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double et = std::exp(t);
        const double denom = 1.0 - (1.0 - p) * et;

        if (denom <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return (p * et) / denom;
    }

    double geometric_cgf_scalar(const double t, const double p) {
        if (!std::isfinite(t) || !std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double et = std::exp(t);
        const double denom = 1.0 - (1.0 - p) * et;

        if (denom <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::log(p) + t - std::log(denom);
    }

    int geometric_sample(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return -1;
        }

        // std::geometric_distribution is undefined at p == 1, which validation admits.
        if (p == 1.0) return 1;

        std::geometric_distribution<int> dist(p);

        // The standard adaptor counts failures before the first success.
        return dist(rng()) + 1;
    }

} // namespace probforge::math
