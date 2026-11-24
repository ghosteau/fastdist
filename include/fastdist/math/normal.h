/**
 * @file normal.h
 * @brief Normal (Gaussian) distribution on the real line.
 *
 *     f(x) = exp(-(x - mu)^2 / (2 sigma^2)) / (sigma sqrt(2 pi))
 *
 * @par Contract
 * Every function returns NaN unless @p sigma > 0 and all arguments are finite.
 * Batch forms write @p n values and evaluate at `x_data[i] + stepSize * i`.
 */
#ifndef NORMAL_H
#define NORMAL_H

#include <cstddef> // size_t

namespace fastdist::math {
    /** @brief Density f(x). */
    double normal_pdf_scalar(double x, double mu, double sigma);

    /**
     * @brief Log density, log f(x) = -z^2/2 - log(sigma) - log(sqrt(2 pi)), z = (x - mu) / sigma.
     *
     * The log of this density, not the log-normal density; for that see normal_log_sample().
     * Preferred over log(normal_pdf_scalar(...)), which underflows to -inf past about 38 sigma.
     */
    double normal_logpdf_scalar(double x, double mu, double sigma);

    /** @brief P(X <= x) = (1 + erf((x - mu) / (sigma sqrt(2)))) / 2. */
    double normal_cdf_scalar(double x, double mu, double sigma);

    /** @brief E[X] = mu. */
    double normal_mean(double mu);

    /** @brief Var[X] = sigma^2. */
    double normal_variance(double sigma);

    /** @brief SD[X] = sigma. */
    double normal_stddev(double sigma);

    /** @brief MGF, M(t) = exp(mu t + sigma^2 t^2 / 2). Finite for all real t. */
    double normal_mgf_scalar(double t, double mu, double sigma);

    /** @brief CGF, K(t) = mu t + sigma^2 t^2 / 2. */
    double normal_cgf_scalar(double t, double mu, double sigma);

    /** @brief One draw from N(mu, sigma^2). */
    double normal_sample(double mu, double sigma);

    /** @brief One draw from the log-normal distribution, exp(X) where X ~ N(mu, sigma^2). */
    double normal_log_sample(double mu, double sigma);

    /** @brief Standard score, (x - mu) / sigma. */
    double z_score(double x, double mu, double sigma);

    /// @name Batch forms
    /// Write @p n values each, evaluating at `x_data[i] + stepSize * i`.
    /// @{
    void normal_pdf_batch(const double* x_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
    void normal_logpdf_batch(const double* x_data, double* output, size_t n, double mu, double sigma,
                             double stepSize = 0);
    void normal_cdf_batch(const double* x_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
    void normal_mgf_batch(const double* t_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
    void normal_cgf_batch(const double* t_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
    /// @}
} // namespace fastdist::math

#endif // NORMAL_H
