// src/api/normal.cpp
#include <probforge/math/normal.h>

extern "C" double fd_normal_pdf(const double x, const double mu, const double sigma) {
    return probforge::math::normal_pdf_scalar(x, mu, sigma);
}

extern "C" double fd_normal_logpdf(const double x, const double mu, const double sigma) {
    return probforge::math::normal_logpdf_scalar(x, mu, sigma);
}

extern "C" double fd_normal_cdf(const double x, const double mu, const double sigma) {
    return probforge::math::normal_cdf_scalar(x, mu, sigma);
}

extern "C" double fd_normal_mean(const double mu) { return probforge::math::normal_mean(mu); }

extern "C" double fd_normal_variance(const double sigma) { return probforge::math::normal_variance(sigma); }

extern "C" double fd_normal_stddev(const double sigma) { return probforge::math::normal_stddev(sigma); }

extern "C" double fd_normal_mgf(const double t, const double mu, const double sigma) {
    return probforge::math::normal_mgf_scalar(t, mu, sigma);
}

extern "C" double fd_normal_cgf(const double t, const double mu, const double sigma) {
    return probforge::math::normal_cgf_scalar(t, mu, sigma);
}

extern "C" double fd_normal_sample(const double mu, const double sigma) {
    return probforge::math::normal_sample(mu, sigma);
}

extern "C" double fd_normal_log_sample(const double mu, const double sigma) {
    return probforge::math::normal_log_sample(mu, sigma);
}

extern "C" double fd_z_score(const double x, const double mu, const double sigma) {
    return probforge::math::z_score(x, mu, sigma);
}
