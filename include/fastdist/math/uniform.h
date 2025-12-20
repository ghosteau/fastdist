// Header file for continuous uniform distribution functions
#ifndef UNIFORM_H
#define UNIFORM_H

// Note: Uniform files all by default refer to continuous uniform distribution
// Continuous uniform distribution is continuous, so we use PDF instead of PMF
namespace fastdist::math {
    // Computes the probability density function (PDF) of the continuous uniform distribution
    double uniform_pdf_scalar(double x, double a, double b);
    // Computes the cumulative density function (CDF) of the continuous uniform distribution
    double uniform_cdf_scalar(double x, double a, double b);
    // Computes the mean of the continuous uniform distribution
    double uniform_mean(double a, double b);
    // Computes the variance of the continuous uniform distribution
    double uniform_variance(double a, double b);
    // Computes the standard deviation of the continuous uniform distribution
    double uniform_stddev(double a, double b);
} // namespace fastdist::math

#endif // UNIFORM_H
