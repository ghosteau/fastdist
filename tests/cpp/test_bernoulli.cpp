// tests/test_bernoulli.cpp
#include <cassert>
#include <cmath>
#include <fastdist/math/bernoulli.h>
#include <fastdist/math/rng.h>
#include <iostream>

void test_bernoulli() {
    constexpr double tol = 1e-6;
    constexpr double p = 0.3;

    // -------------------------
    // MGF tests
    // -------------------------
    {
        assert(std::abs(fastdist::math::bernoulli_mgf_scalar(0.0, p) - 1.0) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM =
                (fastdist::math::bernoulli_mgf_scalar(dt, p) - fastdist::math::bernoulli_mgf_scalar(-dt, p)) / (2 * dt);

        assert(std::abs(dM - fastdist::math::bernoulli_mean(p)) < tol);
    }

    // -------------------------
    // CGF tests
    // -------------------------
    {
        assert(std::abs(fastdist::math::bernoulli_cgf_scalar(0.0, p)) < tol);

        const double dt = 1e-6;
        const double dK =
                (fastdist::math::bernoulli_cgf_scalar(dt, p) - fastdist::math::bernoulli_cgf_scalar(-dt, p)) / (2 * dt);

        assert(std::abs(dK - fastdist::math::bernoulli_mean(p)) < tol);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        // Seeded, so this block is deterministic: it either always passes or
        // always fails on a given toolchain, never intermittently. Tolerances
        // below are ~5x the estimator standard error (SE(mean) = 9.2e-4, SE(var) = 3.7e-4 at this N).
        fastdist::math::seed_rng(1001);

        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            double x = fastdist::math::bernoulli_sample(p);
            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - fastdist::math::bernoulli_mean(p)) < 0.005);
        assert(std::abs(var - fastdist::math::bernoulli_variance(p)) < 0.002);
    }

    std::cout << "Bernoulli tests passed\n";
}
