#include <fastdist/math/uniform.h>

extern "C" double fd_uniform_pdf(double x, double a, double b) { return fastdist::math::uniform_pdf_scalar(x, a, b); }

extern "C" double fd_uniform_cdf(double x, double a, double b) { return fastdist::math::uniform_cdf_scalar(x, a, b); }

extern "C" double fd_uniform_mean(double a, double b) { return fastdist::math::uniform_mean(a, b); }

extern "C" double fd_uniform_variance(double a, double b) { return fastdist::math::uniform_variance(a, b); }

extern "C" double fd_uniform_stddev(double a, double b) { return fastdist::math::uniform_stddev(a, b); }
