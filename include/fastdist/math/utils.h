// Other mathematical utility functions included in the library
#ifndef UTILS_H
#define UTILS_H

namespace fastdist::math {
    // Computes the Chebyshev bound: P(|X - mean| >= k) <= variance / (k^2)
    double chebyshev_bound(double variance, double k);
    // Computes posterior P(A|B) = P(B|A) * P(A) / P(B)
    double bayes_rule(double p_B_given_A, double p_A, double p_B);
    // Sigmoid function: 1 / (1 + exp(-x))
    double sigmoid(double x);
    // Logit function: log(p / (1 - p))
    double logit(double p);
    // Euclidean distance (1D)
    double euclidean_distance(double x, double y);
    // Manhattan distance (1D)
    double manhattan_distance(double x, double y);
    // Coefficient of variation: stddev / |mean|
    double coefficient_of_variation(double mean, double stddev);
    // Population covariance given E[XY], E[X], E[Y]
    double covariance(double mean_x, double mean_y, double E_xy);
} // namespace fastdist::math

#endif // UTILS_H
