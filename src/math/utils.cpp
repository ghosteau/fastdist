// Function declarations for extra library math utility functions
#include "fastdist/math/utils.h"
#include <cmath>
#include <limits>

namespace fastdist::math {

    // Calculates the Chebyshev-Bienaymé inequality bound
    // Definition: P(|X - mean| >= k) <= variance / (k^2)
    // You can also subtract this result from 1 to get the lower bound
    double chebyshev_bound(const double variance, const double k) {
        // k must be positive
        if (!std::isfinite(variance) || variance < 0.0 || !std::isfinite(k) || k <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return variance / (k * k);
    }

} // namespace fastdist::math
