// src/api/exponential.cpp
#include <fastdist/math/exponential.h>

extern "C" double fd_exponential_pdf(double x, double lambda) {
    return fastdist::math::exponential_pdf_scalar(x, lambda);
}

extern "C" double fd_exponential_cdf(double x, double lambda) {
    return fastdist::math::exponential_cdf_scalar(x, lambda);
}

extern "C" double fd_exponential_mean(double lambda) { return fastdist::math::exponential_mean(lambda); }

extern "C" double fd_exponential_variance(double lambda) { return fastdist::math::exponential_variance(lambda); }

extern "C" double fd_exponential_stddev(double lambda) { return fastdist::math::exponential_stddev(lambda); }
