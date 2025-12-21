// Other mathematical utility functions included in the library
#ifndef UTILS_H
#define UTILS_H

namespace fastdist::math {
    // Computes the Chebyshev bound: P(|X - mean| >= k) <= variance / (k^2)
    double chebyshev_bound(double variance, double k);
} // namespace fastdist::math

#endif // UTILS_H
