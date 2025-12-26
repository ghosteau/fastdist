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
    // Calculates n choose k ==> C(n,k)
    double choose(unsigned int n, unsigned int k);
    // Calculates n permute k ==> P(n,k)
    double permutation(unsigned int n, unsigned int k);
    // Calculates n! (factorial)
    double factorial(unsigned int n);
    // Calculates gamma function x
    double gamma(double x);
    // Calculates log gamma function at x
    double log_gamma(double x);
    // Calculates binomial theorem expansion
    double binomial(unsigned int n, double a, double b);
} // namespace fastdist::math

#endif // UTILS_H
