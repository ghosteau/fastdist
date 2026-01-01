// Header file for Negative Binomial distribution functions
#ifndef NEGATIVE_BINOMIAL_H
#define NEGATIVE_BINOMIAL_H

// Negative binomial distribution is discrete, so we use PMF instead of PDF
namespace fastdist::math {
    // Computes the probability mass function (PMF) of the Negative Binomial distribution
    double negative_binomial_pmf_scalar(int k, int r, double p);
    // Computes the cumulative distribution function (CDF) of the Negative Binomial distribution
    double negative_binomial_cdf_scalar(int k, int r, double p);
    // Computes the mean of the Negative Binomial distribution
    double negative_binomial_mean(int r, double p);
    // Computes the variance of the Negative Binomial distribution
    double negative_binomial_variance(int r, double p);
    // Computes the standard deviation of the Negative Binomial distribution
    double negative_binomial_stddev(int r, double p);
    // Computes the moment generating function (MGF) at point t
    double negative_binomial_mgf_scalar(double t, int r, double p);
    // Computes the cumulant generating function (CGF) at point t
    double negative_binomial_cgf_scalar(double t, int r, double p);
    // Draws a random sample from the Negative Binomial distribution
    int negative_binomial_sample(int r, double p);
} // namespace fastdist::math

#endif // NEGATIVE_BINOMIAL_H
