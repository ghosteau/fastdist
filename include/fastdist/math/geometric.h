/**
 * @file geometric.h
 * @brief Geometric distribution: trials up to and including the first success.
 *
 *     P(X = k) = p (1 - p)^(k-1),  k = 1, 2, 3, ...
 *
 * The support counts trials and starts at 1, so the mean is 1/p. The other
 * convention counts failures from 0: that is negative_binomial.h with r = 1.
 *
 * @par Contract
 * NaN unless p is in (0, 1] and finite. Below the support the PMF and CDF are
 * 0; the sampler returns -1 for invalid @p p.
 */
#ifndef GEOMETRIC_H
#define GEOMETRIC_H

namespace fastdist::math {
    /** @brief P(X = k) = p (1 - p)^(k-1). */
    double geometric_pmf_scalar(int k, double p);

    /** @brief P(X <= k) = 1 - (1 - p)^k. */
    double geometric_cdf_scalar(int k, double p);

    /** @brief E[X] = 1/p. */
    double geometric_mean(double p);

    /** @brief Var[X] = (1 - p) / p^2. */
    double geometric_variance(double p);

    /** @brief SD[X] = sqrt(1 - p) / p. */
    double geometric_stddev(double p);

    /** @brief MGF, M(t) = p e^t / (1 - (1 - p) e^t). NaN once t >= -log(1 - p), where it diverges. */
    double geometric_mgf_scalar(double t, double p);

    /** @brief CGF, K(t) = log(p) + t - log(1 - (1 - p) e^t). Same domain as the MGF. */
    double geometric_cgf_scalar(double t, double p);

    /** @brief One draw: the index of the first success. */
    int geometric_sample(double p);
} // namespace fastdist::math

#endif // GEOMETRIC_H
