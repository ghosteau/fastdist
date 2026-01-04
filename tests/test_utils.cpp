#include <cassert>
#include <cmath>
#include <fastdist/math/utils.h>
#include <iostream>
#include <vector>

void test_utils() {
    std::cout << "Running math utils tests...\n";

    constexpr double tol = 1e-12;

    // -------------------------
    // Chebyshev bound
    // -------------------------
    {
        constexpr double var = 4.0;
        constexpr double k = 2.0;
        const double bound = fastdist::math::chebyshev_bound(var, k);
        assert(std::abs(bound - var / (k * k)) < tol);
    }

    // -------------------------
    // Bayes rule
    // -------------------------
    {
        constexpr double p_B_given_A = 0.8;
        constexpr double p_A = 0.3;
        constexpr double p_B = 0.5;
        const double result = fastdist::math::bayes_rule(p_B_given_A, p_A, p_B);
        assert(std::abs(result - (p_B_given_A * p_A / p_B)) < tol);
    }

    // -------------------------
    // Law of Total Probability
    // -------------------------
    {
        const std::vector pB_given_A = {0.2, 0.5};
        const std::vector pA = {0.4, 0.6};
        double total_prob = fastdist::math::law_of_total_probability(pB_given_A, pA);
        assert(std::abs(total_prob - (0.2 * 0.4 + 0.5 * 0.6)) < tol);
    }

    // -------------------------
    // Sigmoid / Logit
    // -------------------------
    {
        constexpr double x = 1.0;
        const double sig = fastdist::math::sigmoid(x);
        assert(std::abs(sig - 1.0 / (1.0 + std::exp(-x))) < tol);

        double p = 0.7;
        const double logit_val = fastdist::math::logit(p);
        assert(std::abs(logit_val - std::log(p / (1 - p))) < tol);
    }

    // -------------------------
    // Distances
    // -------------------------
    {
        const std::vector<double> v1 = {1, 2, 3};
        const std::vector<double> v2 = {4, 5, 6};

        const double eu = fastdist::math::euclidean_distance(v1, v2);
        const double man = fastdist::math::manhattan_distance(v1, v2);
        const double cos = fastdist::math::cosine_similarity(v1, v2);

        assert(std::abs(eu - std::sqrt(27)) < tol);
        assert(std::abs(man - 9) < tol);
        assert(std::abs(cos - (32.0 / (std::sqrt(14) * std::sqrt(77)))) < tol);
    }

    // -------------------------
    // Coefficient of Variation / Covariance
    // -------------------------
    {
        constexpr double mean = 5.0;
        constexpr double stddev = 2.0;
        assert(std::abs(fastdist::math::coefficient_of_variation(mean, stddev) - 0.4) < tol);

        constexpr double mean_x = 2.0, mean_y = 3.0, E_xy = 8.0;
        assert(std::abs(fastdist::math::covariance(mean_x, mean_y, E_xy) - (8.0 - 2.0 * 3.0)) < tol);
    }

    // -------------------------
    // Combinatorics
    // -------------------------
    {
        assert(fastdist::math::choose(5, 2) == 10.0);
        assert(fastdist::math::permutation(5, 2) == 20.0);
        assert(fastdist::math::factorial(5) == 120.0);

        assert(std::abs(fastdist::math::gamma(6.0) - 120.0) < tol);
        assert(std::abs(fastdist::math::log_gamma(6.0) - std::log(120.0)) < tol);

        const double binomial_exp = fastdist::math::binomial(2, 1, 2); // (1+2)^2 = 9
        assert(std::abs(binomial_exp - 9.0) < tol);
    }

    std::cout << "All math utils tests passed!\n";
}
