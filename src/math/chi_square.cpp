// Chi-square distribution, delegated to Gamma(k/2, 2).
// See include/fastdist/math/chi_square.h.
#include <cmath>
#include <fastdist/math/chi_square.h>
#include <fastdist/math/gamma.h>
#include <limits>
#include <random>

namespace fastdist::math {

    double chi_square_pdf_scalar(const double x, const double k) {
        if (!std::isfinite(x) || !std::isfinite(k) || k <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN(); // invalid params or non-finite x
        }
        if (x < 0.0) return 0.0;

        return gamma_pdf_scalar(x, k / 2.0, 2.0);
    }

    double chi_square_cdf_scalar(const double x, const double k) {
        if (!std::isfinite(k) || k <= 0.0) return std::numeric_limits<double>::quiet_NaN(); // invalid params
        if (x < 0.0) return 0.0;

        return gamma_cdf_scalar(x, k / 2.0, 2.0);
    }


    double chi_square_mean(const double k) {
        if (!std::isfinite(k) || k <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        return k;
    }

    double chi_square_variance(const double k) {
        if (!std::isfinite(k) || k <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        return 2.0 * k;
    }

    double chi_square_stddev(const double k) { return std::sqrt(chi_square_variance(k)); }

    double chi_square_mgf_scalar(const double t, const double k) {
        if (!std::isfinite(t) || !std::isfinite(k) || k <= 0.0 || t >= 0.5)
            return std::numeric_limits<double>::quiet_NaN();
        return std::pow(1.0 - 2.0 * t, -k / 2.0);
    }

    double chi_square_cgf_scalar(const double t, const double k) {
        if (!std::isfinite(t) || !std::isfinite(k) || k <= 0.0 || t >= 0.5)
            return std::numeric_limits<double>::quiet_NaN();
        return -0.5 * k * std::log(1.0 - 2.0 * t);
    }

    double chi_square_sample(const double k) {
        if (!std::isfinite(k) || k <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        return gamma_sample(k / 2.0, 2.0);
    }

} // namespace fastdist::math
