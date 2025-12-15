// Header file for normal distribution functions
#ifndef NORMAL_H
#define NORMAL_H

namespace fastdist::math {
    // Computes the probability density function (PDF) of the normal distribution
    double normal_pdf_scalar(double x, double mu, double sigma);
    // Computes the cumulative distribution function (CDF) of the normal distribution
    double normal_cdf_scalar(double x, double mu, double sigma);
    // Computes the mean of the normal distribution
    double normal_mean(double mu);
    // Computes the variance of the normal distribution
    double normal_variance(double sigma);
    // Computes the standard deviation of the normal distribution
    double normal_stddev(double sigma);
} // namespace fastdist::math

#endif //NORMAL_H
