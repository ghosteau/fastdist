// Header file for normal distribution functions
#ifndef NORMAL_H
#define NORMAL_H

namespace fastdist::math {
    // Macros for mathematical constants in normal calculations
    constexpr double SQRT_2PI = 2.506628274631000502415765284811;
    constexpr double LOG_SQRT_2PI = 0.91893853320467274178;

    // Computes the probability density function (PDF) of the normal distribution
    double normal_pdf_scalar(double x, double mu, double sigma);
    // Computes the probability density function (PDF) of the log-normal distribution
    double normal_logpdf_scalar(double x, double mu, double sigma);
    // Computes the cumulative distribution function (CDF) of the normal distribution
    double normal_cdf_scalar(double x, double mu, double sigma);
    // Computes the mean of the normal distribution
    double normal_mean(double mu);
    // Computes the variance of the normal distribution
    double normal_variance(double sigma);
    // Computes the standard deviation of the normal distribution
    double normal_stddev(double sigma);
} // namespace fastdist::math

#endif // NORMAL_H
