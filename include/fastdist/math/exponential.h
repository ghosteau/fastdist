// Header file for exponential distribution functions
#ifndef EXPONENTIAL_H
#define EXPONENTIAL_H

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
} // namespace fastdist::math

#endif // EXPONENTIAL_H
