// Unit tests for Binomial distribution
#include <cassert>
#include <cmath>
#include <fastdist/math/rng.h>
#include <iostream>

#include <fastdist/math/binomial.h>

void test_binomial() {
    std::cout << "Running binomial tests...\n";

    constexpr double tol = 1e-6;
    constexpr int n = 10;
    constexpr double p = 0.3;

    // -------------------------
    // PMF / CDF tests
    // -------------------------
    {
        const double pmf3 = fastdist::math::binomial_pmf_scalar(3, n, p);
        assert(std::abs(pmf3 - 0.266827932) < tol);

        const double cdf2 = fastdist::math::binomial_cdf_scalar(2, n, p);
        assert(std::abs(cdf2 - 0.382782786) < tol);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        assert(std::abs(fastdist::math::binomial_mean(n, p) - n * p) < tol);
        assert(std::abs(fastdist::math::binomial_variance(n, p) - n * p * (1.0 - p)) < tol);
        assert(std::abs(fastdist::math::binomial_stddev(n, p) - std::sqrt(n * p * (1.0 - p))) < tol);
    }

    // -------------------------
    // MGF tests
    // -------------------------
    {
        // M(0) = 1
        assert(std::abs(fastdist::math::binomial_mgf_scalar(0.0, n, p) - 1.0) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM =
                (fastdist::math::binomial_mgf_scalar(dt, n, p) - fastdist::math::binomial_mgf_scalar(-dt, n, p)) /
                (2.0 * dt);

        assert(std::abs(dM - fastdist::math::binomial_mean(n, p)) < tol);
    }

    // -------------------------
    // CGF tests
    // -------------------------
    {
        // K(0) = 0
        assert(std::abs(fastdist::math::binomial_cgf_scalar(0.0, n, p)) < tol);

        // K'(0) = mean
        constexpr double dt = 1e-6;
        const double dK =
                (fastdist::math::binomial_cgf_scalar(dt, n, p) - fastdist::math::binomial_cgf_scalar(-dt, n, p)) /
                (2.0 * dt);

        assert(std::abs(dK - fastdist::math::binomial_mean(n, p)) < tol);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        // Seeded, so this block is deterministic: it either always passes or
        // always fails on a given toolchain, never intermittently. Tolerances
        // below are ~5x the estimator standard error (SE(mean) = 2.9e-3, SE(var) = 5.8e-3 at this N).
        fastdist::math::seed_rng(1003);

        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = fastdist::math::binomial_sample(n, p);
            sum += x;
            sumsq += x * x;

            // support check
            assert(x >= 0.0);
            assert(x <= n);
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - fastdist::math::binomial_mean(n, p)) < 0.015);
        assert(std::abs(var - fastdist::math::binomial_variance(n, p)) < 0.03);
    }

    std::cout << "Binomial tests passed.\n";
}
