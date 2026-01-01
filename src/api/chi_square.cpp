// src/api/chi_square.cpp
#include <fastdist/math/chi_square.h>

extern "C" double fd_chi_square_pdf_scalar(const double x, const double k) {
    return fastdist::math::chi_square_pdf_scalar(x, k);
}

extern "C" double fd_chi_square_cdf_scalar(const double x, const double k) {
    return fastdist::math::chi_square_cdf_scalar(x, k);
}

extern "C" double fd_chi_square_mean(const double k) { return fastdist::math::chi_square_mean(k); }

extern "C" double fd_chi_square_variance(const double k) { return fastdist::math::chi_square_variance(k); }

extern "C" double fd_chi_square_stddev(const double k) { return fastdist::math::chi_square_stddev(k); }

extern "C" double fd_chi_square_mgf_scalar(const double t, const double k) {
    return fastdist::math::chi_square_mgf_scalar(t, k);
}

extern "C" double fd_chi_square_cgf_scalar(const double t, const double k) {
    return fastdist::math::chi_square_cgf_scalar(t, k);
}

extern "C" double fd_chi_square_sample(const double k) { return fastdist::math::chi_square_sample(k); }
