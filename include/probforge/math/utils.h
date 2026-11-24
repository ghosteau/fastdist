/**
 * @file utils.h
 * @brief Statistical and combinatorial helpers that belong to no single distribution.
 *
 * @par Contract
 * Every function returns NaN for a non-finite argument or one outside its
 * stated domain. Sequence functions also return NaN for mismatched or empty
 * inputs. The combinatorial functions take unsigned arguments and return
 * +infinity once the result exceeds a double rather than wrapping.
 */
#ifndef UTILS_H
#define UTILS_H

#include <cstddef> // size_t
#include <vector>

namespace probforge::math {
    /// @name Probability identities
    /// @{

    /**
     * @brief Chebyshev-Bienaymé bound, variance / k^2, for P(|X - mean| >= k).
     *
     * @param variance Must be non-negative.
     * @param k Deviation in absolute units, not multiples of sigma; must be > 0.
     * @return The bound, which may exceed 1 for small k -- it is an inequality,
     *         not a probability, and is not clamped.
     */
    double chebyshev_bound(double variance, double k);

    /** @brief Bayes' rule, P(A|B) = P(B|A) P(A) / P(B). All three must be probabilities, with P(B) != 0. */
    double bayes_rule(double p_B_given_A, double p_A, double p_B);

    /** @brief Law of total probability, P(B) = sum_i P(B|A_i) P(A_i), over @p n pairs. */
    double law_of_total_probability(const double* probs_B_given_A, const double* probs_A, size_t n);

    /** @brief Law of total probability over two equal-length vectors. Bounds-safe overload for binding code. */
    double law_of_total_probability(const std::vector<double>& probs_B_given_A, const std::vector<double>& probs_A);

    /// @}
    /// @name Link functions
    /// @{

    /** @brief Logistic sigmoid, 1 / (1 + e^(-x)). Branches on the sign of x so neither tail overflows. */
    double sigmoid(double x);

    /** @brief Logit, log(p / (1 - p)), the inverse of sigmoid(). Requires p in (0, 1) strictly. */
    double logit(double p);

    /// @}
    /// @name Vector metrics
    /// @{

    /** @brief Euclidean (L2) distance between two equal-length vectors. */
    double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y);

    /** @brief Manhattan (L1) distance between two equal-length vectors. */
    double manhattan_distance(const std::vector<double>& x, const std::vector<double>& y);

    /** @brief Cosine similarity, (x.y) / (|x| |y|). NaN if either vector has zero norm. */
    double cosine_similarity(const std::vector<double>& x, const std::vector<double>& y);

    /// @}
    /// @name Summary statistics
    /// @{

    /** @brief Coefficient of variation, stddev / |mean|. Divides by the magnitude, so the sign of the mean
     *         does not flip the result. Requires mean != 0 and stddev >= 0. */
    double coefficient_of_variation(double mean, double stddev);

    /** @brief Population covariance from its moments, E[XY] - E[X] E[Y]. */
    double covariance(double mean_x, double mean_y, double E_xy);

    /// @}
    /// @name Combinatorics
    /// @{

    /**
     * @brief Binomial coefficient C(n, k).
     *
     * Multiplicative recurrence, taking the smaller of k and n-k by symmetry.
     * Every partial product is itself a binomial coefficient, so each division
     * is exact while the result stays within 2^53.
     *
     * @return 0 for k > n, or +infinity once C(n, k) exceeds a double.
     */
    double choose(unsigned int n, unsigned int k);

    /** @brief Permutation count P(n, k) = n! / (n-k)!. Returns 0 for k > n, +infinity on overflow. */
    double permutation(unsigned int n, unsigned int k);

    /** @brief n!. Returns +infinity above 170, the largest factorial a double can hold. */
    double factorial(unsigned int n);

    /** @brief Gamma(x), via std::tgamma. Overflows between x = 171.6 and 171.7, so Gamma(171) = 170! is
     *         the largest representable integer case; prefer log_gamma() beyond it. */
    double gamma(double x);

    /** @brief log Gamma(x), via std::lgamma. Stays finite far beyond where gamma() overflows. */
    double log_gamma(double x);

    /** @brief Binomial expansion of (a + b)^n as sum_k C(n,k) a^(n-k) b^k. */
    double binomial(unsigned int n, double a, double b);

    /// @}
    /// @name Batch forms
    /// Elementwise over @p n values; a non-finite input yields NaN in its own slot.
    /// @{
    void sigmoid_cpu(const double* x_data, double* output, size_t n);
    void logit_cpu(const double* p_data, double* output, size_t n);
    /// @}
} // namespace probforge::math

#endif // UTILS_H
