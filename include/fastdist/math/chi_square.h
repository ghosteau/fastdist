/**
 * @file chi_square.h
 * @brief Chi-square distribution with k degrees of freedom.
 *
 *     f(x) = x^(k/2-1) e^(-x/2) / (2^(k/2) Gamma(k/2)),  x >= 0
 *
 * This is Gamma with shape k/2 and scale 2, and is implemented by delegating to
 * gamma.h. k is taken as a double so non-integer degrees of freedom work.
 *
 * @par Contract
 * Every function returns NaN unless k > 0 and all arguments are finite. Below
 * the support the PDF and CDF are 0.
 */
#ifndef CHI_SQUARE_H
#define CHI_SQUARE_H

namespace fastdist::math {
    /** @brief Density f(x), via gamma_pdf_scalar(x, k/2, 2). */
    double chi_square_pdf_scalar(double x, double k);

    /** @brief P(X <= x), via gamma_cdf_scalar(x, k/2, 2). */
    double chi_square_cdf_scalar(double x, double k);

    /** @brief E[X] = k. */
    double chi_square_mean(double k);

    /** @brief Var[X] = 2k. */
    double chi_square_variance(double k);

    /** @brief SD[X] = sqrt(2k). */
    double chi_square_stddev(double k);

    /** @brief MGF, M(t) = (1 - 2t)^(-k/2). NaN once t >= 1/2, where it diverges. */
    double chi_square_mgf_scalar(double t, double k);

    /** @brief CGF, K(t) = -(k/2) log(1 - 2t). Computed directly, so large k cannot overflow it. */
    double chi_square_cgf_scalar(double t, double k);

    /** @brief One draw from Chi-square(k). */
    double chi_square_sample(double k);
} // namespace fastdist::math

#endif // CHI_SQUARE_H
