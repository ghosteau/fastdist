/**
 * @file uniform.h
 * @brief Continuous uniform distribution on [a, b].
 *
 *     f(x) = 1 / (b - a),  a <= x <= b
 *
 * See discrete_uniform.h for the integer-valued case.
 *
 * @par Contract
 * Every function returns NaN unless a < b and all arguments are finite; the
 * bounds must be strictly ordered, since a == b has no density. Outside [a, b]
 * the PDF is 0. Batch forms write @p n values and evaluate at
 * `x_data[i] + stepSize * i`.
 */
#ifndef UNIFORM_H
#define UNIFORM_H

#include <cstddef> // size_t

namespace probforge::math {
    /** @brief Density f(x) = 1 / (b - a) on the support, else 0. */
    double uniform_pdf_scalar(double x, double a, double b);

    /** @brief P(X <= x) = (x - a) / (b - a), clamped to [0, 1]. */
    double uniform_cdf_scalar(double x, double a, double b);

    /** @brief E[X] = (a + b) / 2. */
    double uniform_mean(double a, double b);

    /** @brief Var[X] = (b - a)^2 / 12. */
    double uniform_variance(double a, double b);

    /** @brief SD[X] = (b - a) / sqrt(12). */
    double uniform_stddev(double a, double b);

    /** @brief MGF, M(t) = (e^(bt) - e^(at)) / (t (b - a)); M(0) = 1. */
    double uniform_mgf_scalar(double t, double a, double b);

    /** @brief CGF, K(t) = log M(t). */
    double uniform_cgf_scalar(double t, double a, double b);

    /** @brief One draw, uniform on [a, b). */
    double uniform_sample(double a, double b);

    /// @name Batch forms
    /// Write @p n values each, evaluating at `x_data[i] + stepSize * i`.
    /// @{
    void uniform_pdf_batch(const double* x_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
    void uniform_cdf_batch(const double* x_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
    void uniform_mgf_batch(const double* t_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
    void uniform_cgf_batch(const double* t_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
    /// @}
} // namespace probforge::math

#endif // UNIFORM_H
