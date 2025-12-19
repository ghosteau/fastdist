// Header file for binomial distribution functions
#ifndef BINOMIAL_H
#define BINOMIAL_H

// Binomial distribution is discrete, so we use PMF instead of PDF
namespace fastdist::math {
    // Computes the probability mass function (PMF) of the binomial distribution
    double binomial_pmf_scalar(int x, int n, double p);
    // Computes the log probability mass function (log PMF) of the binomial distribution
    double binomial_logpmf_scalar(int x, int n, double p);
    // Computes the cumulative distribution function (CDF) of the binomial distribution
    double binomial_cdf_scalar(int x, int n, double p);
    // Computes the mean of the binomial distribution
    double binomial_mean(int n, double p);
    // Computes the variance of the binomial distribution
    double binomial_variance(int n, double p);
    // Computes the standard deviation of the binomial distribution
    double binomial_stddev(int n, double p);
} // namespace fastdist::math

#endif // BINOMIAL_H
