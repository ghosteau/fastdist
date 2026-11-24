/**
 * @file poisson.h
 * @brief Poisson distribution: event counts, support {0, 1, 2, ...}.
 *
 *     P(X = k) = lambda^k e^(-lambda) / k!
 *
 * @par Contract
 * Every function returns NaN unless lambda > 0 and all arguments are finite.
 * x is taken as a double and is 0 outside the support or at a non-integer
 * point. poisson_sample() returns -1 for invalid input, which the support
 * cannot produce. Batch forms write @p n values and evaluate at
 * `x_data[i] + stepSize * i`.
 */
#ifndef POISSON_H
#define POISSON_H

#include <cstddef> // size_t

namespace probforge::math {
    /**
     * @brief P(X = k).
     *
     * Evaluated in log space as k log(lambda) - lambda - log(k!). Forming
     * lambda^k / k! directly overflows both terms for modest k even where their
     * ratio is an ordinary number.
     */
    double poisson_pmf_scalar(double x, double lambda);

    /** @brief P(X <= floor(x)), summed by PMF recurrence. */
    double poisson_cdf_scalar(double x, double lambda);

    /** @brief E[X] = lambda. */
    double poisson_mean(double lambda);

    /** @brief Var[X] = lambda. */
    double poisson_variance(double lambda);

    /** @brief SD[X] = sqrt(lambda). */
    double poisson_stddev(double lambda);

    /** @brief MGF, M(t) = exp(lambda (e^t - 1)). Overflows for moderate t; prefer the CGF. */
    double poisson_mgf_scalar(double t, double lambda);

    /** @brief CGF, K(t) = lambda (e^t - 1). */
    double poisson_cgf_scalar(double t, double lambda);

    /** @brief One draw from Poisson(lambda). */
    int poisson_sample(double lambda);

    /// @name Batch forms
    /// Write @p n values each, evaluating at `x_data[i] + stepSize * i`.
    /// @{
    void poisson_pmf_batch(const double* x_data, double* output, size_t n, double lambda, int stepSize = 0);
    void poisson_cdf_batch(const double* x_data, double* output, size_t n, double lambda, int stepSize = 0);
    void poisson_mgf_batch(const double* t_data, double* output, size_t n, double lambda, int stepSize = 0);
    void poisson_cgf_batch(const double* t_data, double* output, size_t n, double lambda, int stepSize = 0);
    /// @}
} // namespace probforge::math

#endif // POISSON_H
