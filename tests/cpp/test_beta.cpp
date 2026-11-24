// Unit tests for Beta distribution
#include <cassert>
#include <cmath>
#include <iostream>
#include <probforge/math/rng.h>

#include <probforge/math/beta.h>

void test_beta() {
    std::cout << "Running Beta distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double alpha = 2.0;
    constexpr double beta = 5.0;

    // -------------------------
    // PDF tests
    // f(x) = x^(a-1) (1-x)^(b-1) / B(a,b)
    // -------------------------
    {
        const double x = 0.4;

        const double pdf = probforge::math::beta_pdf_scalar(x, alpha, beta);

        const double B = std::tgamma(alpha) * std::tgamma(beta) / std::tgamma(alpha + beta);

        const double expected = std::pow(x, alpha - 1.0) * std::pow(1.0 - x, beta - 1.0) / B;

        assert(std::abs(pdf - expected) < tol);

        // Outside support
        assert(probforge::math::beta_pdf_scalar(-0.1, alpha, beta) == 0.0);
        assert(probforge::math::beta_pdf_scalar(1.1, alpha, beta) == 0.0);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = probforge::math::beta_mean(alpha, beta);
        const double var = probforge::math::beta_variance(alpha, beta);
        const double std = probforge::math::beta_stddev(alpha, beta);

        const double expected_mean = alpha / (alpha + beta);
        const double expected_var = (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0));

        assert(std::abs(mean - expected_mean) < tol);
        assert(std::abs(var - expected_var) < tol);
        assert(std::abs(std - std::sqrt(expected_var)) < tol);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        // Seeded, so this block is deterministic: it either always passes or
        // always fails on a given toolchain, never intermittently. Tolerances
        // below are ~5x the estimator standard error (SE(mean) = 3.2e-4, SE(var) = 7.0e-5 at this N).
        probforge::math::seed_rng(1002);

        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = probforge::math::beta_sample(alpha, beta);

            // support check
            assert(x >= 0.0);
            assert(x <= 1.0);

            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - probforge::math::beta_mean(alpha, beta)) < 0.002);
        assert(std::abs(var - probforge::math::beta_variance(alpha, beta)) < 4e-4);
    }

    std::cout << "Beta distribution tests passed!\n";
}
