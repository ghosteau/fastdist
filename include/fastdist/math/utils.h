// Other mathematical utility functions included in the library
#ifndef UTILS_H
#define UTILS_H

#include <vector>

namespace fastdist::math {
    // Computes the Chebyshev bound: P(|X - mean| >= k) <= variance / (k^2)
    double chebyshev_bound(double variance, double k);
    // Computes posterior P(A|B) = P(B|A) * P(A) / P(B)
    double bayes_rule(double p_B_given_A, double p_A, double p_B);
    // Computes P(B) = Σ P(B|A_i) * P(A_i) being defined as the Law of Total Probability
    double law_of_total_probability(const double* probs_B_given_A, const double* probs_A, size_t n);
    // Safe C++ wrapper using std::vector for Law of Total Probability
    double law_of_total_probability(const std::vector<double>& probs_B_given_A, const std::vector<double>& probs_A);
    // Sigmoid function: 1 / (1 + exp(-x))
    double sigmoid(double x);
    // Logit function: log(p / (1 - p))
    double logit(double p);
    // Euclidean distance (n-D)
    double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y);
    // Manhattan distance (n-D)
    double manhattan_distance(const std::vector<double>& x, const std::vector<double>& y);
    // Cosine similarity metric calculation (n-D)
    double cosine_similarity(const std::vector<double>& x, const std::vector<double>& y);
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
