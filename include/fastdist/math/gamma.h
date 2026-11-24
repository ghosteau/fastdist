/**
 * @file gamma.h
 * @brief Gamma distribution, shape-scale parameterization.
 *
 *     f(x) = x^(alpha-1) e^(-x/theta) / (Gamma(alpha) theta^alpha),  x >= 0
 *
 * alpha is the shape and theta the scale, so the mean is alpha*theta. Note that
 * the rate parameterization uses 1/theta. Chi-square is the special case
 * alpha = k/2, theta = 2; see chi_square.h.
 *
 * @par Contract
 * Every function returns NaN unless alpha > 0, theta > 0 and all arguments are
 * finite. Below the support the PDF is 0, and the CDF rejects a negative x
 * rather than returning 0.
 */
#ifndef GAMMA_H
#define GAMMA_H

namespace fastdist::math {
    /**
     * @brief Density f(x).
     *
     * Evaluated in log space: Gamma(alpha) and theta^alpha each overflow a
     * double once alpha passes about 171, long before the density does.
     */
    double gamma_pdf_scalar(double x, double alpha, double theta);

    /**
     * @brief P(X <= x), the regularized lower incomplete gamma P(alpha, x/theta).
     *
     * Dispatches between the power series (fast below x = alpha+1) and the
     * modified-Lentz continued fraction (fast above it), per Numerical Recipes
     * 6.2. Both are evaluated in log space.
     */
    double gamma_cdf_scalar(double x, double alpha, double theta);

    /** @brief E[X] = alpha theta. */
    double gamma_mean(double alpha, double theta);

    /** @brief Var[X] = alpha theta^2. */
    double gamma_variance(double alpha, double theta);

    /** @brief SD[X] = theta sqrt(alpha). */
    double gamma_stddev(double alpha, double theta);

    /** @brief MGF, M(t) = (1 - theta t)^(-alpha). NaN once t >= 1/theta, where it diverges. */
    double gamma_mgf_scalar(double t, double alpha, double theta);

    /**
     * @brief CGF, K(t) = -alpha log(1 - theta t).
     *
     * Computed from the formula, not as log(MGF): the MGF overflows for large
     * alpha while the CGF stays an ordinary number.
     */
    double gamma_cgf_scalar(double t, double alpha, double theta);

    /** @brief One draw from Gamma(alpha, theta). */
    double gamma_sample(double alpha, double theta);
} // namespace fastdist::math

#endif // GAMMA_H
