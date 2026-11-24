/**
 * @file bernoulli.h
 * @brief Bernoulli distribution: one trial, discrete, support {0, 1}.
 *
 *     P(X = 1) = p,  P(X = 0) = 1 - p
 *
 * @par Contract
 * Every function returns NaN unless p is in [0, 1] and finite. Outside the
 * support the PMF is 0. bernoulli_sample() returns -1 for invalid @p p, which
 * the support cannot produce. Batch forms write @p n values and evaluate at
 * `k_data[i] + stepSize * i`.
 */
#ifndef BERNOULLI_H
#define BERNOULLI_H

#include <cstddef> // size_t

namespace probforge::math {
    /** @brief P(X = k), i.e. p for k = 1 and 1 - p for k = 0. */
    double bernoulli_pmf_scalar(int k, double p);

    /** @brief P(X <= k). */
    double bernoulli_cdf_scalar(int k, double p);

    /** @brief E[X] = p. */
    double bernoulli_mean(double p);

    /** @brief Var[X] = p(1 - p). */
    double bernoulli_variance(double p);

    /** @brief SD[X] = sqrt(p(1 - p)). */
    double bernoulli_stddev(double p);

    /** @brief MGF, M(t) = (1 - p) + p e^t. */
    double bernoulli_mgf_scalar(double t, double p);

    /** @brief CGF, K(t) = log((1 - p) + p e^t). */
    double bernoulli_cgf_scalar(double t, double p);

    /** @brief One draw: 1 with probability p, else 0. */
    int bernoulli_sample(double p);

    /// @name Batch forms
    /// Write @p n values each, evaluating at `k_data[i] + stepSize * i`.
    /// @{
    void bernoulli_pmf_batch(const int* k_data, double* output, size_t n, double p, int stepSize);
    void bernoulli_cdf_batch(const int* k_data, double* output, size_t n, double p, int stepSize);
    void bernoulli_mgf_batch(const double* t_data, double* output, size_t n, double p, int stepSize);
    void bernoulli_cgf_batch(const double* t_data, double* output, size_t n, double p, int stepSize);
    /// @}
} // namespace probforge::math

#endif // BERNOULLI_H
