// src/api/exponential.cpp
#include <fastdist/math/exponential.h>

extern "C" double fd_exponential_pdf(const double x, const double lambda) {
    return fastdist::math::exponential_pdf_scalar(x, lambda);
}

extern "C" double fd_exponential_cdf(const double x, const double lambda) {
    return fastdist::math::exponential_cdf_scalar(x, lambda);
}

extern "C" double fd_exponential_mean(const double lambda) { return fastdist::math::exponential_mean(lambda); }

extern "C" double fd_exponential_variance(const double lambda) { return fastdist::math::exponential_variance(lambda); }

extern "C" double fd_exponential_stddev(const double lambda) { return fastdist::math::exponential_stddev(lambda); }

extern "C" double fd_exponential_mgf(const double t, const double lambda) {
    return fastdist::math::exponential_mgf_scalar(t, lambda);
}

extern "C" double fd_exponential_cgf(const double t, const double lambda) {
    return fastdist::math::exponential_cgf_scalar(t, lambda);
}

extern "C" double fd_exponential_sample(const double lambda) { return fastdist::math::exponential_sample(lambda); }
