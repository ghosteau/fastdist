// Header file for Bernoulli distribution functions
#ifndef BERNOULLI_H
#define BERNOULLI_H

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
} // namespace fastdist::math

#endif //BERNOULLI_H
