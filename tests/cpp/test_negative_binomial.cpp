// Unit tests for Negative Binomial distribution
#include <cassert>
#include <cmath>
#include <iostream>
#include <probforge/math/rng.h>

#include <probforge/math/negative_binomial.h>

void test_negative_binomial() {
    std::cout << "Running Negative Binomial distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr int r = 4; // number of successes
    constexpr double p = 0.3; // success probability

    // -------------------------
    // PMF / CDF tests
    // -------------------------
    {
        constexpr int k = 3;

        const double pmf = probforge::math::negative_binomial_pmf_scalar(k, r, p);

        const double expected =
                std::tgamma(k + r) / (std::tgamma(r) * std::tgamma(k + 1.0)) * std::pow(1.0 - p, k) * std::pow(p, r);

        assert(std::abs(pmf - expected) < tol);

        // Outside support
        assert(probforge::math::negative_binomial_pmf_scalar(-1, r, p) == 0.0);

        // CDF sanity check
        double sum = 0.0;
        for (int i = 0; i <= k; ++i) {
            sum += probforge::math::negative_binomial_pmf_scalar(i, r, p);
        }

        const double cdf = probforge::math::negative_binomial_cdf_scalar(k, r, p);

        assert(std::abs(cdf - sum) < tol);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = probforge::math::negative_binomial_mean(r, p);
        const double var = probforge::math::negative_binomial_variance(r, p);
        const double std = probforge::math::negative_binomial_stddev(r, p);

        const double expected_mean = r * (1.0 - p) / p;
        const double expected_var = r * (1.0 - p) / (p * p);

        assert(std::abs(mean - expected_mean) < tol);
        assert(std::abs(var - expected_var) < tol);
        assert(std::abs(std - std::sqrt(expected_var)) < tol);
    }

    // -------------------------
    // MGF / CGF tests
    // -------------------------
    {
        const double t = 0.25;

        const double mgf = probforge::math::negative_binomial_mgf_scalar(t, r, p);

        const double expected_mgf = std::pow(p / (1.0 - (1.0 - p) * std::exp(t)), r);

        assert(std::abs(mgf - expected_mgf) < tol);

        const double cgf = probforge::math::negative_binomial_cgf_scalar(t, r, p);

        assert(std::abs(cgf - std::log(expected_mgf)) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM = (probforge::math::negative_binomial_mgf_scalar(dt, r, p) -
                           probforge::math::negative_binomial_mgf_scalar(-dt, r, p)) /
                          (2.0 * dt);

        assert(std::abs(dM - probforge::math::negative_binomial_mean(r, p)) < 1e-6);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        // Seeded, so this block is deterministic: it either always passes or
        // always fails on a given toolchain, never intermittently. Tolerances
        // below are ~5x the estimator standard error (SE(mean) = 1.1e-2, SE(var) = 1.2e-1 at this N).
        probforge::math::seed_rng(1009);

        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const int x = probforge::math::negative_binomial_sample(r, p);

            // support check
            assert(x >= 0);

            sum += x;
            sumsq += static_cast<double>(x) * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - probforge::math::negative_binomial_mean(r, p)) < 0.075);

        assert(std::abs(var - probforge::math::negative_binomial_variance(r, p)) < 0.75);
    }

    std::cout << "Negative Binomial distribution tests passed!\n";
}
