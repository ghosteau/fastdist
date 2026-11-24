/**
 * @file binomial.h
 * @brief Binomial distribution: successes in n independent trials, support {0, ..., n}.
 *
 *     P(X = k) = C(n, k) p^k (1 - p)^(n-k)
 *
 * For sampling *without* replacement, where trials are not independent, see
 * hypergeometric.h.
 *
 * @par Contract
 * Every function returns NaN unless p is in [0, 1] and finite and n >= 0.
 * Outside the support the PMF is 0 and the log PMF is -inf. binomial_sample()
 * returns -1 for invalid input, which the support cannot produce.
 */
#ifndef BINOMIAL_H
#define BINOMIAL_H

namespace probforge::math {
    /**
     * @brief log P(X = k).
     *
     * Evaluated through lgamma: forming C(n, k) from factorials overflows once
     * n passes 170, long before the PMF itself does.
     */
    double binomial_logpmf_scalar(int x, int n, double p);

    /** @brief P(X = k). Exponentiates binomial_logpmf_scalar(), which is what keeps large n in range. */
    double binomial_pmf_scalar(int x, int n, double p);

    /** @brief P(X <= k), summed by PMF recurrence. */
    double binomial_cdf_scalar(int x, int n, double p);

    /** @brief E[X] = n p. */
    double binomial_mean(int n, double p);

    /** @brief Var[X] = n p (1 - p). */
    double binomial_variance(int n, double p);

    /** @brief SD[X] = sqrt(n p (1 - p)). */
    double binomial_stddev(int n, double p);

    /** @brief MGF, M(t) = ((1 - p) + p e^t)^n. */
    double binomial_mgf_scalar(double t, int n, double p);

    /** @brief CGF, K(t) = n log((1 - p) + p e^t). */
    double binomial_cgf_scalar(double t, int n, double p);

    /** @brief One draw from Binomial(n, p). */
    int binomial_sample(int n, double p);
} // namespace probforge::math

#endif // BINOMIAL_H
