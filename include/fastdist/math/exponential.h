// Header file for exponential distribution functions
#ifndef EXPONENTIAL_H
#define EXPONENTIAL_H

#include <cstdio> // For size_t

namespace fastdist::math {
    // Computes the probability density function (PDF) of the exponential distribution
    double exponential_pdf_scalar(double x, double lambda);
    // Computes the cumulative distribution function (CDF) of the exponential distribution
    double exponential_cdf_scalar(double x, double lambda);
    // Computes the mean of the exponential distribution
    double exponential_mean(double lambda);
    // Computes the variance of the exponential distribution
    double exponential_variance(double lambda);
    // Computes the standard deviation of the exponential distribution
    double exponential_stddev(double lambda);
    // Computes exponential MGF at point t
    double exponential_mgf_scalar(double t, double lambda);
    // Computes exponential CGF at point t
    double exponential_cgf_scalar(double t, double lambda);
    // Computes random sample from exponential distribution
    double exponential_sample(double lambda);

    // Batch Functions
    void exponential_pdf_batch(const double* x_data, double* output, size_t n, double lambda, double stepSize);
    void exponential_cdf_batch(const double* x_data, double* output, size_t n, double lambda, double stepSize);
    void exponential_mgf_batch(const double* t_data, double* output, size_t n, double lambda, double stepSize);
    void exponential_cgf_batch(const double* t_data, double* output, size_t n, double lambda, double stepSize);
} // namespace fastdist::math

#endif // EXPONENTIAL_H
