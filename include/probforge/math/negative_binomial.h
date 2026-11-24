/**
 * @file negative_binomial.h
 * @brief Negative binomial distribution: failures before the r-th success.
 *
 *     P(X = k) = C(k + r - 1, k) p^r (1 - p)^k,  k = 0, 1, 2, ...
 *
 * k counts failures, r successes, p the success probability. With r = 1 this
 * is the geometric distribution counted from 0, one less than geometric.h.
 *
 * @par Contract
 * NaN unless p is in (0, 1] and finite and r > 0; at p = 1 the distribution
 * collapses onto k = 0, which is accepted throughout. Below the support the
 * PMF and CDF are 0; the sampler returns -1 for invalid input.
 */
#ifndef NEGATIVE_BINOMIAL_H
#define NEGATIVE_BINOMIAL_H

namespace probforge::math {
    /**
     * @brief P(X = k).
     *
     * Evaluated in log space: C(k + r - 1, k) overflows from factorials once
     * k + r - 1 passes 170, while the PMF stays an ordinary number.
     */
    double negative_binomial_pmf_scalar(int k, int r, double p);

    /** @brief P(X <= k), summed by PMF recurrence. */
    double negative_binomial_cdf_scalar(int k, int r, double p);

    /** @brief E[X] = r (1 - p) / p. */
    double negative_binomial_mean(int r, double p);

    /** @brief Var[X] = r (1 - p) / p^2. */
    double negative_binomial_variance(int r, double p);

    /** @brief SD[X] = sqrt(r (1 - p)) / p. */
    double negative_binomial_stddev(int r, double p);

    /** @brief MGF, M(t) = (p / (1 - (1 - p) e^t))^r. NaN once (1 - p) e^t >= 1, where it diverges. */
    double negative_binomial_mgf_scalar(double t, int r, double p);

    /**
     * @brief CGF, K(t) = r (log p - log(1 - (1 - p) e^t)).
     *
     * Computed from the formula, not as log(MGF): the MGF overflows for large
     * r while the CGF stays an ordinary number.
     */
    double negative_binomial_cgf_scalar(double t, int r, double p);

    /** @brief One draw: the number of failures before the r-th success. */
    int negative_binomial_sample(int r, double p);
} // namespace probforge::math

#endif // NEGATIVE_BINOMIAL_H
