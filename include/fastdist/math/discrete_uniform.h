// Header file for discrete uniform distribution functions
#ifndef DISCRETE_UNIFORM_H
#define DISCRETE_UNIFORM_H

// Discrete uniform distribution is discrete, so we use PMF instead of PDF
namespace fastdist::math {
    // Computes the probability mass function (PMF) of the discrete uniform distribution
    double discrete_uniform_pmf_scalar(int x, int a, int b);
    // Computes the cumulative mass function (CMF) of the discrete uniform distribution
    double discrete_uniform_cdf_scalar(int x, int a, int b);
    // Computes the mean of the discrete uniform distribution
    double discrete_uniform_mean(int a, int b);
    // Computes the variance of the discrete uniform distribution
    double discrete_uniform_variance(int a, int b);
    // Computes the standard deviation of the discrete uniform distribution
    double discrete_uniform_stddev(int a, int b);
    // Computes discrete uniform MGF at point t
    double discrete_uniform_mgf_scalar(double t, int a, int b);
    // Computes discrete uniform CGF at point t
    double discrete_uniform_cgf_scalar(double t, int a, int b);
    // Computes random sample from discrete uniform distribution
    int discrete_uniform_sample(int a, int b);
} // namespace fastdist::math

#endif // DISCRETE_UNIFORM_H
