/**
 * @file beta.h
 * @brief Beta distribution on [0, 1].
 *
 *     f(x) = x^(alpha-1) (1-x)^(beta-1) / B(alpha, beta)
 *
 * @par Contract
 * Every function returns NaN unless alpha > 0, beta > 0 and all arguments are
 * finite. Outside [0, 1] the PDF is 0.
 *
 * @note No MGF or CGF is provided. The Beta MGF is a confluent hypergeometric
 * function with no elementary closed form, and the finite-sum trick available
 * for bounded discrete supports does not apply to a continuous one.
 */
#ifndef BETA_H
#define BETA_H

namespace probforge::math {
    /**
     * @brief Density f(x).
     *
     * Evaluated in log space: Gamma(alpha) and Gamma(beta) overflow once either
     * shape passes about 171, long before the density does.
     */
    double beta_pdf_scalar(double x, double alpha, double beta);

    /**
     * @brief P(X <= x), the regularized incomplete beta I_x(alpha, beta).
     *
     * Modified-Lentz continued fraction (Numerical Recipes 6.4) with the
     * reflection I_x(a,b) = 1 - I_(1-x)(b,a), which maps every input onto the
     * side where the fraction converges quickly. Clamped to [0, 1], since the
     * reflection subtracts nearby quantities in the upper tail.
     */
    double beta_cdf_scalar(double x, double alpha, double beta);

    /** @brief E[X] = alpha / (alpha + beta). */
    double beta_mean(double alpha, double beta);

    /** @brief Var[X] = alpha beta / ((alpha+beta)^2 (alpha+beta+1)). */
    double beta_variance(double alpha, double beta);

    /** @brief SD[X], the square root of beta_variance(). */
    double beta_stddev(double alpha, double beta);

    /** @brief One draw, formed as A / (A + B) with A ~ Gamma(alpha, 1) and B ~ Gamma(beta, 1). */
    double beta_sample(double alpha, double beta);
} // namespace probforge::math

#endif // BETA_H
