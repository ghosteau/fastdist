// Unit tests for Exponential distribution
#include <cassert>
#include <cmath>
#include <iostream>
#include <probforge/math/rng.h>

#include <probforge/math/exponential.h>

void test_exponential() {
    std::cout << "Running exponential distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double lambda = 2.0; // rate parameter

    // -------------------------
    // PDF / CDF tests
    // -------------------------
    {
        const double pdf0 = probforge::math::exponential_pdf_scalar(0.0, lambda);
        assert(std::abs(pdf0 - lambda) < tol);

        const double pdf1 = probforge::math::exponential_pdf_scalar(1.0, lambda);
        assert(std::abs(pdf1 - lambda * std::exp(-lambda)) < tol);

        const double pdf_neg = probforge::math::exponential_pdf_scalar(-1.0, lambda);
        assert(pdf_neg == 0.0);

        const double cdf0 = probforge::math::exponential_cdf_scalar(0.0, lambda);
        assert(std::abs(cdf0 - 0.0) < tol);

        const double cdf1 = probforge::math::exponential_cdf_scalar(1.0, lambda);
        assert(std::abs(cdf1 - (1.0 - std::exp(-lambda))) < tol);

        const double cdf_neg = probforge::math::exponential_cdf_scalar(-1.0, lambda);
        assert(cdf_neg == 0.0);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = probforge::math::exponential_mean(lambda);
        const double var = probforge::math::exponential_variance(lambda);
        const double stddev = probforge::math::exponential_stddev(lambda);

        assert(std::abs(mean - 1.0 / lambda) < tol);
        assert(std::abs(var - 1.0 / (lambda * lambda)) < tol);
        assert(std::abs(stddev - 1.0 / lambda) < tol);
    }

    // -------------------------
    // MGF tests
    // M(t) = λ / (λ - t), t < λ
    // -------------------------
    {
        assert(std::abs(probforge::math::exponential_mgf_scalar(0.0, lambda) - 1.0) < tol);

        constexpr double t = 0.5;
        constexpr double expected = lambda / (lambda - t);
        const double mgf = probforge::math::exponential_mgf_scalar(t, lambda);

        assert(std::abs(mgf - expected) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM = (probforge::math::exponential_mgf_scalar(dt, lambda) -
                           probforge::math::exponential_mgf_scalar(-dt, lambda)) /
                          (2.0 * dt);

        assert(std::abs(dM - probforge::math::exponential_mean(lambda)) < 1e-6);
    }

    // -------------------------
    // CGF tests
    // K(t) = -log(1 - t/λ)
    // -------------------------
    {
        assert(std::abs(probforge::math::exponential_cgf_scalar(0.0, lambda)) < tol);

        constexpr double t = 0.5;
        const double expected = -std::log(1.0 - t / lambda);
        const double cgf = probforge::math::exponential_cgf_scalar(t, lambda);

        assert(std::abs(cgf - expected) < tol);

        // K'(0) = mean
        constexpr double dt = 1e-6;
        const double dK = (probforge::math::exponential_cgf_scalar(dt, lambda) -
                           probforge::math::exponential_cgf_scalar(-dt, lambda)) /
                          (2.0 * dt);

        assert(std::abs(dK - probforge::math::exponential_mean(lambda)) < 1e-6);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        // Seeded, so this block is deterministic: it either always passes or
        // always fails on a given toolchain, never intermittently. Tolerances
        // below are ~5x the estimator standard error (SE(mean) = 1.0e-3, SE(var) = 1.4e-3 at this N).
        probforge::math::seed_rng(1006);

        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = probforge::math::exponential_sample(lambda);

            // support check
            assert(x >= 0.0);

            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - probforge::math::exponential_mean(lambda)) < 0.005);
        assert(std::abs(var - probforge::math::exponential_variance(lambda)) < 0.0075);
    }

    std::cout << "Exponential distribution tests passed.\n";
}
