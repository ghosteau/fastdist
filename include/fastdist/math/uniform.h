// Header file for continuous uniform distribution functions
#ifndef UNIFORM_H
#define UNIFORM_H

#include <cstddef> // size_t

// Continuous uniform distribution on [a, b]. See discrete_uniform.h for the
// integer-valued case.
namespace fastdist::math {
    // Computes the probability density function (PDF) of the continuous uniform distribution
    double uniform_pdf_scalar(double x, double a, double b);
    // Computes the cumulative distribution function (CDF) of the continuous uniform distribution
    double uniform_cdf_scalar(double x, double a, double b);
    // Computes the mean of the continuous uniform distribution
    double uniform_mean(double a, double b);
    // Computes the variance of the continuous uniform distribution
    double uniform_variance(double a, double b);
    // Computes the standard deviation of the continuous uniform distribution
    double uniform_stddev(double a, double b);
    // Computes the moment-generating function (MGF) of the continuous uniform distribution
    double uniform_mgf_scalar(double t, double a, double b);
    // Computes the cumulant-generating function (CGF) of the continuous uniform distribution
    double uniform_cgf_scalar(double t, double a, double b);
    // Computes a random sample from the continuous uniform distribution
    double uniform_sample(double a, double b);

    // Batch functions: output[i] = f(x_data[i] + stepSize * i) for i in [0, n).
    // A stepSize of 0 evaluates x_data as given. Invalid parameters make every
    // output NaN.
    void uniform_pdf_batch(const double* x_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
    void uniform_cdf_batch(const double* x_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
    void uniform_mgf_batch(const double* t_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
    void uniform_cgf_batch(const double* t_data, double* output, size_t n, double a, double b, double stepSize = 0.0);
} // namespace fastdist::math

#endif // UNIFORM_H
