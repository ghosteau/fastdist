// Header file for Bernoulli distribution functions
#ifndef BERNOULLI_H
#define BERNOULLI_H

#include <cstdio> // For size_t

// Bernoulli distribution is discrete, so we use PMF instead of PDF
namespace fastdist::math {
    // Computes the probability mass function (PMF) of the Bernoulli distribution
    double bernoulli_pmf_scalar(int k, double p);
    // Computes the cumulative distribution function (CDF) of the Bernoulli distribution
    double bernoulli_cdf_scalar(int k, double p);
    // Computes the mean of the Bernoulli distribution
    double bernoulli_mean(double p);
    // Computes the variance of the Bernoulli distribution
    double bernoulli_variance(double p);
    // Computes the standard deviation of the Bernoulli distribution
    double bernoulli_stddev(double p);
    // Computes Bernoulli MGF at point t
    double bernoulli_mgf_scalar(double t, double p);
    // Computes Bernoulli CGF at point t
    double bernoulli_cgf_scalar(double t, double p);
    // Computes random sample from Bernoulli distribution
    int bernoulli_sample(double p);

    // Batch Functions
    void bernoulli_pmf_batch(const int* k_data, double* output, size_t n, double p, int stepSize);
    void bernoulli_cdf_batch(const int* k_data, double* output, size_t n, double p, int stepSize);
    void bernoulli_mgf_batch(const double* t_data, double* output, size_t n, double p, int stepSize);
    void bernoulli_cgf_batch(const double* t_data, double* output, size_t n, double p, int stepSize);

} // namespace fastdist::math

#endif // BERNOULLI_H
