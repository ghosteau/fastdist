// src/api/utils.cpp
#include "fastdist/math/utils.h"

extern "C" double fd_chebyshev_bound(const double variance, const double k) {
    return fastdist::math::chebyshev_bound(variance, k);
}
