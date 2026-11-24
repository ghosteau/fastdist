/**
 * @file exponential.h
 * @brief Exponential distribution: waiting time, rate parameterization.
 *
 *     f(x) = lambda e^(-lambda x),  x >= 0
 *
 * lambda is a rate, not a scale, so the mean is 1/lambda. This is Gamma with
 * shape 1 and scale 1/lambda; see gamma.h.
 *
 * @par Contract
 * Every function returns NaN unless lambda > 0 and all arguments are finite.
 * Below the support the PDF and CDF are 0. Batch forms write @p n values and
 * evaluate at `x_data[i] + stepSize * i`.
 */
#ifndef EXPONENTIAL_H
#define EXPONENTIAL_H

#include <cstddef> // size_t

namespace fastdist::math {
    /** @brief Density f(x) = lambda e^(-lambda x). */
    double exponential_pdf_scalar(double x, double lambda);

    /** @brief P(X <= x) = 1 - e^(-lambda x), evaluated so small lambda*x does not cancel. */
    double exponential_cdf_scalar(double x, double lambda);

    /** @brief E[X] = 1 / lambda. */
    double exponential_mean(double lambda);

    /** @brief Var[X] = 1 / lambda^2. */
    double exponential_variance(double lambda);

    /** @brief SD[X] = 1 / lambda. */
    double exponential_stddev(double lambda);

    /** @brief MGF, M(t) = lambda / (lambda - t). NaN once t >= lambda, where it diverges. */
    double exponential_mgf_scalar(double t, double lambda);

    /** @brief CGF, K(t) = log(lambda) - log(lambda - t). Same domain as the MGF. */
    double exponential_cgf_scalar(double t, double lambda);

    /** @brief One draw from Exponential(lambda). */
    double exponential_sample(double lambda);

    /// @name Batch forms
    /// Write @p n values each, evaluating at `x_data[i] + stepSize * i`.
    /// @{
    void exponential_pdf_batch(const double* x_data, double* output, size_t n, double lambda, double stepSize);
    void exponential_cdf_batch(const double* x_data, double* output, size_t n, double lambda, double stepSize);
    void exponential_mgf_batch(const double* t_data, double* output, size_t n, double lambda, double stepSize);
    void exponential_cgf_batch(const double* t_data, double* output, size_t n, double lambda, double stepSize);
    /// @}
} // namespace fastdist::math

#endif // EXPONENTIAL_H
