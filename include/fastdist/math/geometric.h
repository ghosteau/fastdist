// Header file for geometric distribution functions
#ifndef GEOMETRIC_H
#define GEOMETRIC_H

// Geometric distribution is discrete, so we use PMF instead of PDF
namespace fastdist::math {
    // Computes the probability mass function (PMF) of the geometric distribution
    double geometric_pmf_scalar(int k, double p);
    // Computes the cumulative mass function (CMF) of the geometric distribution
    double geometric_cdf_scalar(int k, double p);
    // Computes the mean of the geometric distribution
    double geometric_mean(double p);
    // Computes the variance of the geometric distribution
    double geometric_variance(double p);
    // Computes the standard deviation of the geometric distribution
    double geometric_stddev(double p);
    // Computes geometric MGF at point t
    double geometric_mgf_scalar(double t, double p);
    // Computes geometric CGF at point t
    double geometric_cgf_scalar(double t, double p);
    // Computes random sample from geometric distribution
    int geometric_sample(double p);
} // namespace fastdist::math

#endif // GEOMETRIC_H
