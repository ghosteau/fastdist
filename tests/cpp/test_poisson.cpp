#include <cassert>
#include <cmath>
#include <iostream>
#include <probforge/math/poisson.h>
#include <probforge/math/rng.h>

void test_poisson() {
    std::cout << "Running Poisson distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double lambda = 4.0;

    // -------------------------
    // PMF / logPMF / CDF tests
    // -------------------------
    {
        const double x = 3.0;
        const double pmf = probforge::math::poisson_pmf_scalar(x, lambda);
        const double expected_pmf = std::exp(-lambda) * std::pow(lambda, x) / std::tgamma(x + 1.0);
        assert(std::abs(pmf - expected_pmf) < tol);

        // CDF
        const double cdf = probforge::math::poisson_cdf_scalar(x, lambda);
        double sum = 0.0;
        for (int i = 0; i <= static_cast<int>(x); ++i) {
            sum += probforge::math::poisson_pmf_scalar(i, lambda);
        }
        assert(std::abs(cdf - sum) < tol);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        assert(std::abs(probforge::math::poisson_mean(lambda) - lambda) < tol);
        assert(std::abs(probforge::math::poisson_variance(lambda) - lambda) < tol);
        assert(std::abs(probforge::math::poisson_stddev(lambda) - std::sqrt(lambda)) < tol);
    }

    // -------------------------
    // MGF / CGF tests
    // -------------------------
    {
        const double t = 0.25;
        assert(std::abs(probforge::math::poisson_mgf_scalar(t, lambda) - std::exp(lambda * (std::exp(t) - 1.0))) < tol);
        assert(std::abs(probforge::math::poisson_cgf_scalar(t, lambda) - lambda * (std::exp(t) - 1.0)) < tol);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        // Seeded, so this block is deterministic: it either always passes or
        // always fails on a given toolchain, never intermittently. Tolerances
        // below are ~5x the estimator standard error (SE(mean) = 4.0e-3, SE(var) = 1.2e-2 at this N).
        probforge::math::seed_rng(1010);

        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const int x = probforge::math::poisson_sample(lambda);
            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - lambda) < 0.02);
        assert(std::abs(var - lambda) < 0.075);
    }

    std::cout << "Poisson distribution tests passed!\n";
}
