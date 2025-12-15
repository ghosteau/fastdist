// Header file for poisson distribution functions
#ifndef POISSON_H
#define POISSON_H

// Poisson distribution is discrete, so we use PMF instead of PDF
namespace fastdist::math {
    // Computes the probability mass function (PMF) of the poisson distribution
    double poisson_pmf_scalar(double x, double lambda);
    // Computes the cumulative mass function (CMF) of the poisson distribution
    double poisson_cdf_scalar(double x, double lambda);
    // Computes the mean of the poisson distribution
    double poisson_mean(double lambda);
    // Computes the variance of the poisson distribution
    double poisson_variance(double lambda);
    // Computes the standard deviation of the poisson distribution
    double poisson_stddev(double lambda);
} // namespace fastdist::math

#endif //POISSON_H
