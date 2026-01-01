// Header file for Chi-square distribution functions
#ifndef CHI_SQUARE_H
#define CHI_SQUARE_H

namespace fastdist::math {
    // Computes the PDF of the Chi-square distribution
    double chi_square_pdf_scalar(double x, double k);
    // Computes the CDF of the Chi-square distribution
    double chi_square_cdf_scalar(double x, double k);
    // Computes the mean of the Chi-square distribution
    double chi_square_mean(double k);
    // Computes the variance of the Chi-square distribution
    double chi_square_variance(double k);
    // Computes the standard deviation of the Chi-square distribution
    double chi_square_stddev(double k);
    // Computes the moment generating function (MGF) at point t
    double chi_square_mgf_scalar(double t, double k);
    // Computes the cumulant generating function (CGF) at point t
    double chi_square_cgf_scalar(double t, double k);
    // Draws a random sample from the Chi-square distribution
    double chi_square_sample(double k);
} // namespace fastdist::math

#endif // CHI_SQUARE_H
