/**
 * @file hypergeometric.h
 * @brief Hypergeometric distribution: successes in a draw made without replacement.
 *
 *     P(X = k) = C(K, k) C(N - K, n - k) / C(N, n)
 *
 * Each draw removes an item, so unlike the binomial the trials are not
 * independent and the variance is smaller by the finite population correction.
 * This is the exact null distribution behind Fisher's exact test.
 *
 *     N = population   size of the whole population
 *     K = successes    items in the population that count as a success, K <= N
 *     n = draws        items drawn without replacement, n <= N
 *     k                successes observed among the draws
 *
 * Support: max(0, n - (N - K)) <= k <= min(n, K). It does not always start at
 * zero -- a draw larger than the pool of failures must contain successes.
 *
 * Parameters are named rather than given as SciPy's (M, n, N) triple, whose
 * ordering is easy to transpose silently. SciPy's M is @p population, its n is
 * @p successes and its N is @p draws.
 *
 * @par Contract
 * Every function returns NaN unless all three parameters are non-negative with
 * @p successes <= @p population and @p draws <= @p population; the moments and
 * generating functions additionally require @p population > 0. Outside the
 * support the PMF is 0 and the log PMF is -inf. hypergeometric_sample()
 * returns -1 for invalid input, which the support cannot produce.
 *
 * @par Accuracy
 * The PMF is within 1.4e-12 relative of exact rational arithmetic. That is the
 * floor for a log-space method rather than a loose implementation: lgamma(1001)
 * is about 5912.13, whose ULP is already 9.1e-13, so the log cannot be held
 * more precisely. Reaching machine precision needs Loader's method ("Fast and
 * Accurate Computation of Binomial Probabilities", 2000).
 */
#ifndef HYPERGEOMETRIC_H
#define HYPERGEOMETRIC_H

namespace fastdist::math {
    /** @brief log P(X = k), through lgamma so the coefficients cannot overflow. */
    double hypergeometric_logpmf_scalar(int k, int population, int successes, int draws);

    /** @brief P(X = k). */
    double hypergeometric_pmf_scalar(int k, int population, int successes, int draws);

    /** @brief P(X <= k), summed by PMF recurrence from the bottom of the support. */
    double hypergeometric_cdf_scalar(int k, int population, int successes, int draws);

    /** @brief E[X] = n K / N. */
    double hypergeometric_mean(int population, int successes, int draws);

    /**
     * @brief Var[X] = n (K/N) ((N-K)/N) ((N-n)/(N-1)).
     *
     * The last factor is the finite population correction, and it is what
     * separates this from a binomial variance: it falls to zero when the whole
     * population is drawn. Returns 0 at @p population == 1, where the draw is
     * deterministic and the correction would evaluate 0/0; SciPy reports NaN
     * for that case.
     */
    double hypergeometric_variance(int population, int successes, int draws);

    /** @brief SD[X], the square root of hypergeometric_variance(). */
    double hypergeometric_stddev(int population, int successes, int draws);

    /**
     * @brief MGF, M(t) = sum_k e^(tk) P(X = k).
     *
     * There is no elementary closed form -- it is a Gaussian hypergeometric 2F1
     * -- but the support is finite, so the defining sum is exact. Accumulated by
     * log-sum-exp, which is why the CGF below stays finite where this overflows.
     */
    double hypergeometric_mgf_scalar(double t, int population, int successes, int draws);

    /** @brief CGF, K(t) = log M(t). Finite well past the point where the MGF overflows. */
    double hypergeometric_cgf_scalar(double t, int population, int successes, int draws);

    /**
     * @brief One draw: successes observed in @p draws items taken without replacement.
     *
     * Simulates the draw directly, since there is no
     * std::hypergeometric_distribution: the chance the next item is a success
     * is the proportion of successes among what is left. Exact for any
     * parameters rather than an approximation that degrades at the edges.
     */
    int hypergeometric_sample(int population, int successes, int draws);
} // namespace fastdist::math

#endif // HYPERGEOMETRIC_H
