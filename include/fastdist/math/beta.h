// Header file for Beta distribution functions
#ifndef BETA_H
#define BETA_H

// Note: Beta distribution MGF does not have a simple closed form
namespace fastdist::math {
    // Computes the probability density function (PDF) of the Beta distribution
    double beta_pdf_scalar(double x, double alpha, double beta);
    // Computes the cumulative distribution function (CDF) of the Beta distribution
    double beta_cdf_scalar(double x, double alpha, double beta);
    // Computes the mean of the Beta distribution
    double beta_mean(double alpha, double beta);
    // Computes the variance of the Beta distribution
    double beta_variance(double alpha, double beta);
    // Computes the standard deviation of the Beta distribution
    double beta_stddev(double alpha, double beta);
    // Draws a random sample from the Beta distribution
    double beta_sample(double alpha, double beta);
} // namespace fastdist::math

#endif // BETA_H
