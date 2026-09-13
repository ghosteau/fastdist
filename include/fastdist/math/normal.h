// Header file for normal distribution functions
#ifndef NORMAL_H
#define NORMAL_H

#include <cstddef> // size_t

namespace fastdist::math {
    // Computes the probability density function (PDF) of the normal distribution
    double normal_pdf_scalar(double x, double mu, double sigma);
    // Computes the natural log of the normal PDF (not the log-normal density)
    double normal_logpdf_scalar(double x, double mu, double sigma);
    // Computes the cumulative distribution function (CDF) of the normal distribution
    double normal_cdf_scalar(double x, double mu, double sigma);
    // Computes the mean of the normal distribution
    double normal_mean(double mu);
    // Computes the variance of the normal distribution
    double normal_variance(double sigma);
    // Computes the standard deviation of the normal distribution
    double normal_stddev(double sigma);
    // Computes normal MGF at point t
    double normal_mgf_scalar(double t, double mu, double sigma);
    // Computes normal CGF at point t
    double normal_cgf_scalar(double t, double mu, double sigma);
    // Computes random sample from normal distribution
    double normal_sample(double mu, double sigma);
    // Draws a sample from the log-normal distribution: exp(X), X ~ N(mu, sigma^2)
    double normal_log_sample(double mu, double sigma);
    // Computes the z-score for a given x in the normal distribution
    double z_score(double x, double mu, double sigma);

    // Batch functions: output[i] = f(x_data[i] + stepSize * i) for i in [0, n).
    // A stepSize of 0 evaluates x_data as given. Invalid parameters make every
    // output NaN.
    void normal_pdf_batch(const double* x_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
    void normal_logpdf_batch(const double* x_data, double* output, size_t n, double mu, double sigma,
                             double stepSize = 0);
    void normal_cdf_batch(const double* x_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
    void normal_mgf_batch(const double* t_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
    void normal_cgf_batch(const double* t_data, double* output, size_t n, double mu, double sigma, double stepSize = 0);
} // namespace fastdist::math

#endif // NORMAL_H
