// Function definitions for geometric distribution functions
#include <cmath>
#include <fastdist/math/geometric.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

namespace fastdist::math {

    double geometric_pmf_scalar(const int k, const double p) {
        // p must be in (0,1]
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Geometric is defined for k >= 1
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

        // 1 - (1-p)^k cancels catastrophically for small p: (1-p)^k rounds to a
        // value just under 1 and the leading digits of the difference are lost.
        // At p = 1e-15 that cost three correct digits of a four-digit answer.
        // expm1 of the log form keeps full precision across the whole range.
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

    // M_X(t) = p e^t / (1 - (1 - p)e^t),  t < -log(1 - p)
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

    // X ~ Geometric(p), support {1,2,...}
    int geometric_sample(const double p) {
        if (!std::isfinite(p) || p <= 0.0 || p > 1.0) {
            return -1;
        }

        // std::geometric_distribution is only defined for 0 < p < 1, and the
        // validation above admits p == 1. At p == 1 the first trial always
        // succeeds, so the draw is 1 with certainty.
        if (p == 1.0) return 1;

        std::geometric_distribution<int> dist(p);

        // std::geometric_distribution returns #failures before first success
        return dist(rng()) + 1;
    }

} // namespace fastdist::math
