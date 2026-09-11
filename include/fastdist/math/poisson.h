// Header file for poisson distribution functions
#ifndef POISSON_H
#define POISSON_H

#include <cstddef> // size_t

// Poisson distribution is discrete, so we use PMF instead of PDF
namespace fastdist::math {
    // Computes the probability mass function (PMF) of the poisson distribution
    double poisson_pmf_scalar(double x, double lambda);
    // Computes the cumulative distribution function (CDF) of the poisson distribution
    double poisson_cdf_scalar(double x, double lambda);
    // Computes the mean of the poisson distribution
    double poisson_mean(double lambda);
    // Computes the variance of the poisson distribution
    double poisson_variance(double lambda);
    // Computes the standard deviation of the poisson distribution
    double poisson_stddev(double lambda);
    // Computes the moment-generating function (MGF) of the Poisson distribution
    double poisson_mgf_scalar(double t, double lambda);
    // Computes the cumulant-generating function (CGF) of the Poisson distribution
    double poisson_cgf_scalar(double t, double lambda);
    // Computes a random sample from the Poisson distribution
    int poisson_sample(double lambda);

    // Batch functions: output[i] = f(x_data[i] + stepSize * i) for i in [0, n).
    // A stepSize of 0 evaluates x_data as given. Invalid parameters make every
    // output NaN.
    void poisson_pmf_batch(const double* x_data, double* output, size_t n, double lambda, int stepSize = 0);
    void poisson_cdf_batch(const double* x_data, double* output, size_t n, double lambda, int stepSize = 0);
    void poisson_mgf_batch(const double* t_data, double* output, size_t n, double lambda, int stepSize = 0);
    void poisson_cgf_batch(const double* t_data, double* output, size_t n, double lambda, int stepSize = 0);

} // namespace fastdist::math

#endif // POISSON_H
