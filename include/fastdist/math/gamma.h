// Header file for gamma distribution functions
#ifndef GAMMA_H
#define GAMMA_H

namespace fastdist::math {
    // Macros for mathematical constants in gamma calculations
    constexpr unsigned int MAX_ITER = 100;
    constexpr double EPS = 1e-12;
    constexpr double FPMIN = 1e-30;

    // Computes the probability density function (PDF) of the Gamma distribution
    double gamma_pdf_scalar(double x, double alpha, double theta);
    // Computes the cumulative distribution function (CDF) of the Gamma distribution
    double gamma_cdf_scalar(double x, double alpha, double theta);
    // Computes the mean of the Gamma distribution
    double gamma_mean(double alpha, double theta);
    // Computes the variance of the Gamma distribution
    double gamma_variance(double alpha, double theta);
    // Computes the standard deviation of the Gamma distribution
    double gamma_stddev(double alpha, double theta);
    // Computes the moment generating function (MGF) at point t
    double gamma_mgf_scalar(double t, double alpha, double theta);
    // Computes the cumulant generating function (CGF) at point t
    double gamma_cgf_scalar(double t, double alpha, double theta);
    // Draws a random sample from the Gamma distribution
    double gamma_sample(double alpha, double theta);
} // namespace fastdist::math

#endif // GAMMA_H
